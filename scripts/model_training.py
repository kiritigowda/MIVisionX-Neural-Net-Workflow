import os
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

    def runDocker(self):
        os.system('sudo docker rm -f training')
        os.system('sudo docker run -it -d -v $(pwd):/root/hostDrive/ --name training --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host mivisionx/pytorch-ubuntu-16.04 bash')
        os.system('sudo docker cp rali_training_setup.py training:/')    
        os.system('sudo docker cp %s training:/' % self.datapath)
        self.setupDone = True
        os.system('sudo docker exec -i training bash -c "python3.6 rali_training_setup.py --dataset %s --batch_size %d --epochs %d --path %s"' % (self.dataset_folder, self.batch_size, self.epochs, self.PATH))

    def getEpoch(self):
        with open('result_file.csv', 'r') as f:
            for line in f:
                print(line.split(' ')[0])
