import os
import torch
import argparse
import numpy as np
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from PIL import Image

DATA_BACKEND_CHOICES = ['pytorch']
from amd.rali.plugin.pytorch import RALIClassificationIterator
from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types

def ResNet(device):
    model = models.resnet50().to(device)
    return model

class train_pipeline(Pipeline):
    def __init__(self, data_path, batch_size, num_thread, crop, rali_cpu = True):
        super(train_pipeline, self).__init__(batch_size, num_thread, rali_cpu=rali_cpu)
        world_size = 1
        local_rank = 0
        self.input = ops.FileReader(file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        rali_device = 'cpu' if rali_cpu else 'gpu'
        decoder_device = 'cpu' if rali_cpu else 'mixed'
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                    device_memory_padding=device_memory_padding,
                                                    host_memory_padding=host_memory_padding,
                                                    random_aspect_ratio=[0.8, 1.25],
                                                    random_area=[0.1, 1.0],
                                                    num_attempts=100)
        self.res = ops.Resize(device=rali_device, resize_x=crop, resize_y=crop)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('rali "{0}" variant'.format(rali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]

def get_pytorch_train_loader(data_path, batch_size, num_thread, crop, rali_cpu):
    print("in get_pytorch_train_loader fucntion")   
    pipe_train = train_pipeline(data_path, batch_size, num_thread, crop, rali_cpu)
    pipe_train.build()
    train_loader = RALIClassificationIterator(pipe_train)
    return train_loader


class val_pipeline(Pipeline):
    def __init__(self, data_path, batch_size, num_thread, crop, rali_cpu = True):
        super(val_pipeline, self).__init__(batch_size, num_thread, rali_cpu=rali_cpu)
        world_size = 1
        local_rank = 0
        self.input = ops.FileReader(file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        rali_device = 'cpu' if rali_cpu else 'gpu'
        decoder_device = 'cpu' if rali_cpu else 'mixed'
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.RGB)
        self.res = ops.Resize(device=rali_device, resize_x=256, resize_y=256)
        self.centrecrop = ops.CentreCrop(crop=(crop, crop))
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        print('rali "{0}" variant'.format(rali_device))

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        images = self.centrecrop(images)
        output = self.cmnp(images)
        return [output, self.labels]

def get_pytorch_val_loader(data_path, batch_size, num_thread, crop, rali_cpu):
    print("in get_pytorch_val_loader function")  
    pipe_val = val_pipeline(data_path, batch_size, num_thread, crop, rali_cpu)
    pipe_val.build()
    val_loader = RALIClassificationIterator(pipe_val)
    return val_loader

def train(model, device, train_loader, optimizer, criterion, epoch):
    print("in train fucntion")
    print("epoch:: ",epoch)
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print_interval = 1
        if i % print_interval == (print_interval-1):
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / print_interval))
            running_loss = 0.0

def test(model, device, val_loader, PATH):
    model.load_state_dict(torch.load(PATH))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def main():
    training_device = not args.training_device and torch.cuda.is_available() #checks for rocm installation of pytorch
    device = torch.device("cuda" if training_device else "cpu") #device = GPU in this case
    model = args.model
    PATH = args.path

    dataset_train = args.dataset + '/train'
    if not os.path.exists(dataset_train):
        print("the dataset is invalid, requires train folder")
        exit(0)
    dataset_val = args.dataset + '/val'
    if not os.path.exists(dataset_val):
        print("the dataset is invalid, requires val folder")
        exit(0)
    batch_size = args.batch_size
    epochs = args.epochs
    num_gpu = args.GPU
    input_dims = args.input_dimensions
    rali_cpu = args.rali_cpu
    num_thread = 1
    crop = 224

    model = ResNet(device)
    print(model)

    #training
    train_loader = get_pytorch_train_loader(dataset_train, batch_size, num_thread, crop, rali_cpu)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, criterion, epoch)

    print('Finished Training')
    torch.save(model.state_dict(), PATH)      #save trained model

    #test
    val_loader = get_pytorch_val_loader(dataset_val, batch_size, num_thread, crop, rali_cpu)
    test(model, device, val_loader, PATH)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet50', required=False, help="PyTorch model for training")
    parser.add_argument('--dataset', required=False, help="Dataset with train and val folders")
    parser.add_argument('--batch_size', type=int, required=False, default=32, help="Batch size required for training")
    parser.add_argument('--epochs', type=int, required=False, default=10, help="Number of epochs for training")
    parser.add_argument('--GPU', type=int, required=False,default=1, help="Number of epochs for training")
    parser.add_argument('--input_dimensions', required=False, default='1,3,224,224', help="Model input dimensions")
    parser.add_argument('--rali_cpu', type=bool, required=False, default=True, help="Rali cpu/gpu (true/false)")
    parser.add_argument('--training_device', type=bool, required=False, default=False, help="Use cpu/gpu based training (true/false)")
    parser.add_argument('--path', required=True, help="Path to store trained model")

    args = parser.parse_args()
    main()