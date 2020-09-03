import os
from PyQt5 import QtCore

class modelTraining(QtCore.QObject):
    def __init__(self, model, datapath, PATH, training_device, num_gpu, batch_size, epochs, rali_cpu, input_dims, num_thread, gui, parent=None):
        super(modelTraining, self).__init__(parent)
        self.model = model
        self.datapath = datapath
        self.PATH = PATH
        self.training_device = training_device
        self.num_gpu = num_gpu
        self.batch_size = batch_size
        self.epochs = epochs
        self.rali_cpu = rali_cpu
        self.input_dims = input_dims
        self.num_thread = num_thread
        self.gui = gui

        self.dataset_train = datapath + '/train'
        self.dataset_val = datapath + '/val'

    def runDocker(self):
        os.system('sudo docker rm -f training')
        os.system('sudo docker run -it -d --name training --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host mivisionx/pytorch-ubuntu-16.04 bash')
        os.system('sudo docker cp rali_training_setup.py training:/')    
        os.system('sudo docker cp %s training:/' % self.datapath)
        os.system('sudo docker exec -i training bash -c "python3.6 rali_training_setup.py --dataset %s --batch_size 16 --epochs 10 --path model_rali.pth"' % self.datapath)

    