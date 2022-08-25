import os
import csv
import subprocess
from PyQt5 import QtCore

class modelTraining(QtCore.QObject):
    def __init__(self, model, datapath, PATH, training_device, num_gpu, batch_size, epochs, rocal_cpu, input_dims, num_thread, parent=None):
        super(modelTraining, self).__init__(parent)
        self.model = model
        self.datapath = datapath
        self.dataset_folder = os.path.basename(datapath)
        self.PATH = PATH
        self.training_device = training_device
        self.num_gpu = num_gpu
        self.batch_size = batch_size
        self.epochs = epochs
        self.rocal_cpu = rocal_cpu
        self.input_dims = input_dims
        self.num_thread = num_thread
        self.setupDone = False

    def runDocker(self):
        if (os.path.exists('statistics.csv')):
            os.system('sudo rm statistics.csv')

        print("Training start")
        os.system('sudo docker rm -f training')
        os.system('sudo docker run -it -d -v $(pwd):/workspace/MIVisionX-Neural-Net-Workflow -v %s:/data/imagenet_20dir/ --name training --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host --env DISPLAY=unix$DISPLAY --privileged --volume $XAUTH:/root/.Xauthority --volume /tmp/.X11-unix/:/tmp/.X11-unix lakshmikumar/rocm5.2_pytorch_mivisionx:v0.1 bash' % self.datapath)
        os.system('sudo docker start training')
        os.system('sudo docker exec training bash -c "cd /workspace/MIVisionX/; mkdir build; cd build; cmake -DNEURAL_NET=OFF ../; make -j8; make install"')
        os.system('sudo docker exec training bash -c "cd /workspace/MIVisionX/rocAL/rocAL_pybind/ ; ./run.sh"')
        os.system('sudo docker exec training bash -c "python3 /workspace/MIVisionX-Neural-Net-Workflow/rocal_training_setup.py --dataset /data/imagenet_20dir/ --batch_size %d --epochs %d --path %s"' % (self.batch_size, self.epochs, self.PATH))
        os.system('sudo docker exec training bash -c "exit"')
        self.setupDone = True

    def getValues(self):
        if (os.path.exists('statistics.csv')):
            with open('statistics.csv', 'r') as f:
                for line in f:
                    values = [x.strip() for x in line.split(',')]
                    epoch = values[0]
                    loss = values[1]
                    top1 = values[2]
                    top5 = values[3]

                    return (int)(epoch), (float)(loss), (float)(top1), (float)(top5)
        else:
            return (0,0,0,0)

    def terminate(self):
        print("Terminating...")
        os.system('sudo docker stop training')
        print("Done")

    def isSetupDone(self):
        return self.setupDone