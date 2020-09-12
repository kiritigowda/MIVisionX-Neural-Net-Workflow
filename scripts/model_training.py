import os
import csv
from PyQt5 import QtCore

class modelTraining(QtCore.QObject):
    def __init__(self, model, datapath, PATH, training_device, num_gpu, batch_size, epochs, rali_cpu, input_dims, num_thread, parent=None):
        super(modelTraining, self).__init__(parent)
        self.model = model
        self.datapath = datapath
        self.dataset_folder = os.path.basename(datapath)
        self.PATH = PATH
        self.training_device = training_device
        self.num_gpu = num_gpu
        self.batch_size = batch_size
        self.epochs = epochs
        self.rali_cpu = rali_cpu
        self.input_dims = input_dims
        self.num_thread = num_thread
        self.setupDone = False

    def runDocker(self):
        #os.system('sudo docker rm -f training')
        #os.system('sudo docker run -it -d -v $(pwd):/root/hostDrive/ --name training --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host mivisionx/pytorch-ubuntu-16.04 bash')
        os.system('sudo docker start training')    
        os.system('sudo docker cp %s training:/' % self.datapath)
        os.system('sudo docker exec -i training bash -c "python3.6 rali_training_setup.py --dataset %s --batch_size %d --epochs %d --path %s"' % (self.dataset_folder, self.batch_size, self.epochs, self.PATH))
        self.setupDone = True

    def getValues(self):
        #print ('getting epoch value')
        with open('statistics.csv', 'r') as f:
            for line in f:
                values = [x.strip() for x in line.split(',')]
                epoch = values[0]
                loss = values[1]
                top1 = values[2]
                top5 = values[3]

                return (int)(epoch), (float)(loss), (float)(top1), (float)(top5)

    def terminate(self):
        os.system('sudo docker stop training')

    def isSetupDone(self):
        return self.setupDone