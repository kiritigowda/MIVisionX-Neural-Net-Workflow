import os
import pyqtgraph as pg
import queue
from PyQt5 import QtGui, uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTime, QTimer, QThread
from glob import iglob
from model_training import *

class TrainViewer(QtGui.QMainWindow):
    def __init__(self, model, datapath, PATH, training_device, num_gpu, batch_size, epochs, rali_cpu, input_dims, num_thread, gui, parent):
        super(TrainViewer, self).__init__(parent)
        self.parent = parent

        self.model = model
        self.datapath = datapath
        self.PATH = PATH
        self.training_device = training_device
        self.num_gpu = num_gpu
        self.batch_size = batch_size
        self.epoch = (int)(epochs)
        self.rali_cpu = rali_cpu
        self.input_dims = input_dims
        self.num_thread = num_thread
        self.gui = gui
        self.trainEngine = None
        self.receiver_thread = None
        self.lastEpoch = 0
        self.initEngines()
        
        if self.gui:
            uic.loadUi("train_viewer.ui", self)
            self.model_label.setText("Training model: %s" % model)
            trainpath = datapath + '/train/**'
            self.image_list = [f for f in iglob(trainpath, recursive=True) if os.path.isfile(f) and (f.endswith("JPEG") or f.endswith("PNG") or f.endswith("JPG"))]
            self.it = iter(self.image_list)
            self.model_progressBar.setValue(0)
            self.model_progressBar.setMinimum(0)
            self.model_progressBar.setMaximum(self.epoch)
            self.mProg_label.setText("Epoch %d of %d" % (0, self.epoch))
            self.frameCount = 8

            self.imgCount = self.frameCount
            self.lastIndex = self.frameCount - 1
            self.epochs = [0] 
            self.loss = [4]
            self.top1 = [0]
            self.top5 = [0]

            self.lgraph = None
            self.agraph = None
            self.t1Curve = None
            self.t5Curve= None
            self.legend = None
            #self.imageLength = self.image_frame.Width() - 10
            self.loss_label.setStyleSheet("color: red")
            self.top1_label.setStyleSheet("color: blue")
            self.top5_label.setStyleSheet("color: green")
            self.initGraph()
            self.updateTimer = QTimer()
            self.snoopTimer = QTimer()
            self.updateTimer.timeout.connect(self.update)
            self.updateTimer.timeout.connect(self.showImage)
            self.snoopTimer.timeout.connect(self.getEpochValues)
            self.updateTimer.start(100)
            self.snoopTimer.start(300)
       
    def initGraph(self):
        #self.setStyleSheet("background-color: white")
        self.lgraph = pg.PlotWidget()
        self.lgraph.setLabel('left', 'Loss(%)')
        self.lgraph.setLabel('bottom', 'Epoch(s)')
        self.lgraph.setXRange(0, self.epoch+1, padding=0)
        self.lgraph.setYRange(0, 5.5, padding=0)

        self.agraph = pg.PlotWidget()
        self.agraph.setLabel('left', 'Accuracy(%)')
        self.agraph.setLabel('bottom', 'Epoch(s)')
        self.agraph.setXRange(0, self.epoch+1, padding=0)
        self.agraph.setYRange(0, 105, padding=0)

        pg.setConfigOptions(antialias=True)
        
        self.lgraph.setBackground(None)
        self.lgraph.getAxis('left').setPen(pg.mkPen('k', width=2))
        self.lgraph.getAxis('left').setTextPen(pg.mkPen('k', width=2))
        self.lgraph.getAxis('bottom').setPen(pg.mkPen('k', width=2))
        self.lgraph.getAxis('bottom').setTextPen(pg.mkPen('k', width=2))
        self.agraph.setBackground(None)
        self.agraph.getAxis('left').setPen(pg.mkPen('k', width=2))
        self.agraph.getAxis('left').setTextPen(pg.mkPen('k', width=2))
        self.agraph.getAxis('bottom').setPen(pg.mkPen('k', width=2))        
        self.agraph.getAxis('bottom').setTextPen(pg.mkPen('k', width=2))        
        self.lcurve = self.lgraph.plot(pen=pg.mkPen('r', width=4))
        self.t1Curve = self.agraph.plot(pen=pg.mkPen('b', width=4))
        self.t5Curve = self.agraph.plot(pen=pg.mkPen('g', width=4))
        self.legend = pg.LegendItem(offset=(-20,5), labelTextColor='k')
        self.legend.setParentItem(self.agraph.plotItem)
        self.legend.addItem(self.t1Curve, 'Top1')
        self.legend.addItem(self.t5Curve, 'Top5')
        self.verticalLayout.addWidget(self.lgraph)
        self.verticalLayout_2.addWidget(self.agraph)
    
    def initEngines(self):
        self.receiver_thread = QThread()

        # Creating an object for train
        self.trainEngine = modelTraining(self.model, self.datapath, self.PATH, self.training_device, self.num_gpu, self.batch_size, self.epoch, self.rali_cpu, self.input_dims, self.num_thread)
        
        self.trainEngine.moveToThread(self.receiver_thread)
        self.receiver_thread.started.connect(self.trainEngine.runDocker)
        self.receiver_thread.finished.connect(self.trainEngine.deleteLater)
        self.receiver_thread.start()
        self.receiver_thread.terminate()

    def getEpochValues(self):
        epoch, loss, top1, top5 = self.trainEngine.getValues()
        if epoch is self.epoch:
            self.updateTimer.stop()
            
        if self.lastEpoch is not epoch and epoch is not 0:
            self.mProg_label.setText("Epoch %d of %d" % (epoch, self.epoch))
            self.loss_label.setText("Loss %0.1f%%" % loss)
            self.top1_label.setText("Top1: %0.1f%%" % top1)
            self.top5_label.setText("Top5: %0.1f%%" % top5)
            self.lastEpoch = epoch
            self.model_progressBar.setValue(epoch)
            self.epochs.append(epoch)
            self.loss.append(loss)
            self.top1.append(top1)
            self.top5.append(top5)
            self.plotGraph()

    def showImage(self):
        index = self.imgCount % self.frameCount
        pixmap = QPixmap(next(self.it)).scaled(180, 180)
        self.image_layout.itemAt(index).widget().setPixmap(pixmap)
        self.image_layout.itemAt(index).widget().setStyleSheet("border: 5px solid darkred;");
        self.image_layout.itemAt(self.lastIndex).widget().setStyleSheet("border: 0");
        self.imgCount += 1
        self.lastIndex = index

    def plotGraph(self):
        self.lcurve.setData(x=self.epochs, y=self.loss)
        self.t1Curve.setData(x=self.epochs, y=self.top1)
        self.t5Curve.setData(x=self.epochs, y=self.top5)
    
    # def resizeEvent(self, event):
    #     self.imageLength = self.image_frame.Width() - 10

    def closeEvent(self, event):
        self.trainEngine.terminate()
        #self.receiver_thread.quit()
        self.close()
        exit(0)
