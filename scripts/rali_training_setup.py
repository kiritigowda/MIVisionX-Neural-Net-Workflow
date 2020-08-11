import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

DATA_BACKEND_CHOICES = ['pytorch']
from amd.rali.plugin.pytorch import RALIClassificationIterator
from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types


class train_pipeline(Pipeline):
    def __init__(self, data_path, batch_size, num_thread, crop, rali_cpu = False):
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
    pipe_train = train_pipeline(data_path, batch_size, num_thread, crop, rali_cpu)
    pipe_train.build()
    train_loader = RALIClassificationIterator(pipe_train)
    return train_loader


class val_pipeline(Pipeline):
    def __init__(self, data_path, batch_size, num_thread, crop, rali_cpu = False):
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
    pipe_val = val_pipeline(data_path, batch_size, num_thread, crop, rali_cpu)
    pipe_val.build()
    val_loader = RALIClassificationIterator(pipe_val)
    return val_loader