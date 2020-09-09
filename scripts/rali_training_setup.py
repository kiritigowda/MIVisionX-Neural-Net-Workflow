import os
import torch
import argparse
import csv
import time
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

class Net():
    def __init__(self, device):
        self.device = device

    def ResNet(self):
        model = models.resnet50().to(self.device)
        return model

class trainPipeline(Pipeline):
    def __init__(self, data_path, batch_size, num_thread, crop, rali_cpu = True):
        super(trainPipeline, self).__init__(batch_size, num_thread, rali_cpu=rali_cpu)
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

class trainLoader():
    def __init__(self, data_path, batch_size, num_thread, crop, rali_cpu):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_thread = num_thread
        self.crop = crop
        self.rali_cpu = rali_cpu

    def get_pytorch_train_loader(self):
        print("in get_pytorch_train_loader fucntion")   
        pipe_train = trainPipeline(self.data_path, self.batch_size, self.num_thread, self.crop, self.rali_cpu)
        pipe_train.build()
        train_loader = RALIClassificationIterator(pipe_train)
        return train_loader


class valPipeline(Pipeline):
    def __init__(self, data_path, batch_size, num_thread, crop, rali_cpu = True):
        super(valPipeline, self).__init__(batch_size, num_thread, rali_cpu=rali_cpu)
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

class valLoader():
    def __init__(self, data_path, batch_size, num_thread, crop, rali_cpu):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_thread = num_thread
        self.crop = crop
        self.rali_cpu = rali_cpu

    def get_pytorch_val_loader(self):
        pipe_val = valPipeline(self.data_path, self.batch_size, self.num_thread, self.crop, self.rali_cpu)
        pipe_val.build()
        val_loader = RALIClassificationIterator(pipe_val)
        return val_loader

class AverageMeter(object):
    #Computes and stores the average and current value
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class trainAndTest():
    def __init__(self, net, device, train_loader, val_loader, optimizer, criterion, PATH):
        self.net = net
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.PATH = PATH
        self.results = []
        self.test_acc = 0.0

    def accuracy(self, outputs, labels, topk=(1,5)):
        #Computes the accuracy over the k top predictions for the specified values of k
        with torch.no_grad():
            maxk = max(topk)
            batch_size = labels.size(0)
            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def train(self, epoch):
        print("epoch:: ", epoch)
        running_loss = 0.0
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for i, (inputs, labels) in enumerate(self.train_loader, 0):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # print statistics
            #loss
            running_loss += loss.item()
            losses.update(running_loss, inputs.size(0))

            print_interval = 1
            if i % print_interval == (print_interval-1):
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / print_interval))
                running_loss = 0.0

            #accuracy
            acc1, acc5 = self.accuracy(outputs, labels)
            top1.update(acc1, inputs.size(0))
            top5.update(acc5, inputs.size(0))

            print_interval = 1
            if i % print_interval == (print_interval-1):
                print('[%d, %5d] accuracy1: %.3f' %
                      (epoch + 1, i + 1, acc1 / print_interval))
                print('[%d, %5d] accuracy5: %.3f' %
                      (epoch + 1, i + 1, acc5 / print_interval))

        temp = [epoch, losses.avg, top1.avg.item(), top5.avg.item()]
        return temp
        #self.results.append(temp)
            
    def test(self):
        self.net.load_state_dict(torch.load(self.PATH))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.val_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        self.test_acc = 100 * correct / total
        #print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

    def getLoss(self, epoch):
        return self.results[epoch][1]


    def getTop1(self, epoch):
        return self.results[epoch][2]


    def getTop5(self, epoch):
        return self.results[epoch][3]

def main():
    #initialing parameters
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
    input_dimensions = list(args.input_dimensions.split(","))
    crop = int(input_dimensions[3]) #crop to the width or height of model input_dimensions
    results_file =  'statistics.csv'        	

    #object for class Net
    net_obj = Net(device)
    if model == 'resnet50':
        net = net_obj.ResNet()
        #print(net)

    #train loader
    train_loader_obj = trainLoader(dataset_train, batch_size, num_thread, crop, rali_cpu)
    train_loader = train_loader_obj.get_pytorch_train_loader()

    #print(train_loader)
    #test loader
    val_loader_obj = valLoader(dataset_val, batch_size, num_thread, crop, rali_cpu)
    val_loader = val_loader_obj.get_pytorch_val_loader()
    #print(val_loader)

    optimizer = optim.SGD(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_test_obj = trainAndTest(net, device, train_loader, val_loader, optimizer, criterion, PATH)
    with open(results_file, 'w') as csvfile:
        fieldnames = ['timestamp', 'epoch', 'running_loss', 'top1', 'top5']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'timestamp':0.0, 'epoch':0, 'running_loss':0.0, 'top1':0.0, 'top5':0.0})
    for epoch in range(epochs):
        start_time = time.time()
        results = train_test_obj.train(epoch)
        end_time = time.time()
        time_elapsed = end_time - start_time
        with open(results_file, "r") as infile:
            reader = csv.reader(infile)
            next(reader, None)  # skip the header
            for row in reader:
                old_timestamp = float(row[0])     
        with open(results_file, 'w') as outfile:
            fieldnames = ['timestamp', 'epoch', 'running_loss', 'top1', 'top5']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            new_timestamp = old_timestamp + time_elapsed
            writer.writerow({'timestamp':new_timestamp, 'epoch':results[0], 'running_loss':results[1], 'top1':results[2], 'top5'        :results[3]})
        	
    #print('final results' , train_test_obj.results)
    print('Finished Training')
    torch.save(net.state_dict(), PATH)      #save trained model

    train_test_obj.test()		#validation accuracy
    print('test accuracy' , train_test_obj.test_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet50', required=False, help="PyTorch model for training")
    parser.add_argument('--dataset', required=False, help="Dataset with train and val folders")
    parser.add_argument('--batch_size', type=int, required=False, default=32, help="Batch size required for training")
    parser.add_argument('--epochs', type=int, required=False, default=10, help="Number of epochs for training")
    parser.add_argument('--GPU', type=int, required=False,default=1, help="Number of GPUs for training")
    parser.add_argument('--input_dimensions', required=False, default='1,3,224,224', help="Model input dimensions")
    parser.add_argument('--rali_cpu', type=bool, required=False, default=True, help="Rali cpu/gpu (true/false)")
    parser.add_argument('--training_device', type=bool, required=False, default=False, help="Use cpu/gpu based training (true/false)")
    parser.add_argument('--path', required=True, help="Path to store trained model")

    args = parser.parse_args()
    main()