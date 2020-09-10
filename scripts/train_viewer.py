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
        self.epochs = epochs
        self.rali_cpu = rali_cpu
        self.input_dims = input_dims
        self.num_thread = num_thread
        self.gui = gui
        self.trainEngine = None
        self.receiver_thread = None

        self.initEngines()
        
        if self.gui:
            uic.loadUi("train_viewer.ui", self)
            self.model_label.setText(model)
            trainpath = datapath + '/train/**'
            self.image_list = [f for f in iglob(trainpath, recursive=True) if os.path.isfile(f) and (f.endswith("JPEG") or f.endswith("PNG") or f.endswith("JPG"))]
            self.it = iter(self.image_list)
            self.model_progressBar.setValue(0)
            self.model_progressBar.setMaximum(epochs)
            self.mProg_label.setText("Epoch %d of %d" % (1, self.epochs))
            # for image in self.image_list:
            #     count = count + 1
            #     print (image)
            # print(count)
            # self.container_index = (int)(container_logo)
            # self.cpu_name = cpu_name
            # self.gpu_name = gpu_name
            # self.pauseState = False
            # self.showAug = False
            # self.elapsedTime = QTime.currentTime()
            # self.lastTime = 0
            # self.totalElapsedTime = 0.0
            # self.totalAccuracy = 0
            # self.progIndex = 0
            # self.augIntensity = 0.0
            # self.imgCount = 0
            # self.frameCount = 9
            # self.lastIndex = self.frameCount - 1
            self.x = [0] 
            self.y = [0]
            # self.augAccuracy = []
            # self.totalCurve = None
            # self.augCurve = None
            self.lgraph = None
            self.agraph = None
            self.legend = None
            self.pen = pg.mkPen('w', width=4)
            # self.AMD_Radeon_pixmap = QPixmap("./data/images/AMD_Radeon.png")
            # self.AMD_Radeon_white_pixmap = QPixmap("./data/images/AMD_Radeon-white.png")
            # self.MIVisionX_pixmap = QPixmap("./data/images/MIVisionX-logo.png")
            # self.MIVisionX_white_pixmap = QPixmap("./data/images/MIVisionX-logo-white.png")
            # self.EPYC_pixmap = QPixmap("./data/images/EPYC-blue.png")
            # self.EPYC_white_pixmap = QPixmap("./data/images/EPYC-blue-white.png")
            # self.docker_pixmap = QPixmap("./data/images/Docker.png")
            # self.singularity_pixmap = QPixmap("./data/images/Singularity.png")
            # self.rali_pixmap = QPixmap("./data/images/RALI.png")
            # self.rali_white_pixmap = QPixmap("./data/images/RALI-white.png")
            # self.graph_image_pixmap = QPixmap("./data/images/Graph-image.png")
            # self.graph_image_white_pixmap = QPixmap("./data/images/Graph-image-white.png")
            
            self.initGraph()
            self.updateTimer = QTimer()
            self.snoopTimer = QTimer()
            self.updateTimer.timeout.connect(self.update)
            self.updateTimer.timeout.connect(self.showImage)
            self.snoopTimer.timeout.connect(self.)
            self.updateTimer.start(100)
       
    def initGraph(self):
        self.setStyleSheet("background-color: white")
        self.lgraph = pg.PlotWidget(title="Loss vs Epoch")
        self.lgraph.setLabel('left', 'Loss')
        self.lgraph.setLabel('bottom', 'Epoch(s)')
        self.lgraph.setXRange(0, self.epochs, padding=0)
        self.lgraph.setYRange(0, 100, padding=0)

        self.agraph = pg.PlotWidget(title="Accuracy vs Epoch")
        self.agraph.setLabel('left', 'Accuracy')
        self.agraph.setLabel('bottom', 'Epoch(s)')
        self.agraph.setXRange(0, self.epochs, padding=0)
        self.agraph.setYRange(0, 100, padding=0)

        pg.setConfigOptions(antialias=True)
        
        self.lcurve = self.lgraph.plot(pen=pg.mkPen('b', width=4))
        self.acurve = self.agraph.plot(pen=pg.mkPen('b', width=4))
        self.lgraph.setBackground(None)
        self.lgraph.setMaximumWidth(550)
        self.lgraph.setMaximumHeight(300)
        self.agraph.setBackground(None)
        self.agraph.setMaximumWidth(550)
        self.agraph.setMaximumHeight(300)        
        self.legend = pg.LegendItem(offset=(370,1))
        self.legend.setParentItem(self.lgraph.plotItem)
        self.legend.setParentItem(self.agraph.plotItem)
        self.verticalLayout.addWidget(self.lgraph)
        self.verticalLayout_2.addWidget(self.agraph)

    def initEngines(self):
        self.receiver_thread = QThread()
        # Creating an object for train
        self.trainEngine = modelTraining(self.model, self.datapath, self.PATH, self.training_device, self.num_gpu, self.batch_size, self.epochs, self.rali_cpu, self.input_dims, self.num_thread)
        
        self.trainEngine.moveToThread(self.receiver_thread)
        self.receiver_thread.started.connect(self.trainEngine.runDocker)
        self.receiver_thread.finished.connect(self.trainEngine.deleteLater)
        self.receiver_thread.start()
        self.receiver_thread.terminate()

    # def paintEvent(self, event):
    #     pixmap = QPixmap(next(self.it)).scaled(200,200)
    #     self.image_frame.setPixmap(pixmap)
        #self.showAugImage()
        #self.showImage()
        #self.displayFPS()
        # if self.imgCount == self.total_images:
        #     if self.loop == 'yes':
        #         self.resetViewer()
        #     else:
        #        self.terminate()
    
    def showImage(self):
        pixmap = QPixmap(next(self.it)).scaled(200,200)
        self.image_frame.setPixmap(pixmap)
    # def resetViewer(self):
    #     self.imgCount = 0
    #     del self.x[:]
    #     self.x.append(0)
    #     del self.y[:]
    #     self.y.append(0)
    #     del self.augAccuracy[:]
    #     for augmentation in range(self.batch_size_int):
    #         self.augAccuracy.append([0])

    #     self.lastTime = 0
    #     self.elapsedTime = QTime.currentTime()
    #     self.totalElapsedTime = 0.0
    #     self.progIndex = 0
    #     self.showAug = False
    #     self.lastIndex = self.frameCount - 1
    #     self.totalCurve.clear()
    #     self.augCurve.clear()
    #     self.name_label.setText("Model: %s" % (self.model_name))
    #     self.legend.removeItem(self.lastAugName)

    #def plotGraph(self):
        #self.x.append(epoch)
        #self.y.append(self.totalAccuracy)
        #self.lgraph.setData(x=self.x, y=self.y, pen=self.pen)
        # for augmentation in range(self.batch_size_int):
        #     augStats = self.inferenceEngine.getAugStats(augmentation)
        #     top5 = augStats[1]
        #     mis = augStats[2]
        #     totalCount = top5 + mis
        #     totalAccuracy = (float)(top5) / (totalCount+1) * 100
        #     self.augAccuracy[augmentation].append(totalAccuracy)
    
        # if self.showAug:
        #     self.augCurve.setData(x=self.x, y=self.augAccuracy[self.progIndex], pen=pg.mkPen('b', width=4))

    # def setBackground(self):
    #     if self.dark_checkBox.isChecked():
    #         self.setStyleSheet("background-color: #25232F;")
    #         self.pen = pg.mkPen('w', width=4)
    #         self.graph.setBackground(None)
    #         self.origTitle_label.setStyleSheet("color: #C82327;")
    #         self.controlTitle_label.setStyleSheet("color: #C82327;")
    #         self.progTitle_label.setStyleSheet("color: #C82327;")
    #         self.graphTitle_label.setStyleSheet("color: #C82327;")
    #         self.augTitle_label.setStyleSheet("color: #C82327;")
    #         self.name_label.setStyleSheet("color: white;")
    #         self.dataset_label.setStyleSheet("color: white;")
    #         self.imgProg_label.setStyleSheet("color: white;")
    #         self.fps_label.setStyleSheet("color: #C82327;")
    #         self.dark_checkBox.setStyleSheet("color: white;")
    #         self.verbose_checkBox.setStyleSheet("color: white;")
    #         self.rali_checkBox.setStyleSheet("color: white;")
    #         self.level_label.setStyleSheet("color: white;")
    #         self.low_label.setStyleSheet("color: white;")
    #         self.high_label.setStyleSheet("color: white;")
    #         self.cpu_label.setStyleSheet("color: #C82327;")
    #         self.gpu_label.setStyleSheet("color: #C82327;")
    #         self.cpuName_label.setStyleSheet("color: white;")
    #         self.gpuName_label.setStyleSheet("color: white;")
    #         self.AMD_logo.setPixmap(self.AMD_Radeon_white_pixmap)
    #         if self.rali_checkBox.isChecked():
    #             self.MIVisionX_logo.setPixmap(self.rali_white_pixmap)
    #             self.graph_imageLabel.setPixmap(self.graph_image_white_pixmap)
    #         else:
    #             self.MIVisionX_logo.setPixmap(self.MIVisionX_white_pixmap)
    #         self.EPYC_logo.setPixmap(self.EPYC_white_pixmap)
    #         self.totalCurve.setData(x=self.x, y=self.y, pen=self.pen)
    #     else:
    #         self.setStyleSheet("background-color: white;")
    #         self.pen = pg.mkPen('k', width=4)
    #         self.graph.setBackground(None)
    #         self.origTitle_label.setStyleSheet("color: 0;")
    #         self.controlTitle_label.setStyleSheet("color: 0;")
    #         self.progTitle_label.setStyleSheet("color: 0;")
    #         self.graphTitle_label.setStyleSheet("color: 0;")
    #         self.augTitle_label.setStyleSheet("color: 0;")
    #         self.name_label.setStyleSheet("color: 0;")
    #         self.dataset_label.setStyleSheet("color: 0;")
    #         self.imgProg_label.setStyleSheet("color: 0;")
    #         self.fps_label.setStyleSheet("color: 0;")
    #         self.dark_checkBox.setStyleSheet("color: 0;")
    #         self.verbose_checkBox.setStyleSheet("color: 0;")
    #         self.rali_checkBox.setStyleSheet("color: 0;")
    #         self.level_label.setStyleSheet("color: 0;")
    #         self.low_label.setStyleSheet("color: 0;")
    #         self.high_label.setStyleSheet("color: 0;")
    #         self.cpu_label.setStyleSheet("color: 0;")
    #         self.gpu_label.setStyleSheet("color: 0;")
    #         self.cpuName_label.setStyleSheet("color: 0;")
    #         self.gpuName_label.setStyleSheet("color: 0;")
    #         self.AMD_logo.setPixmap(self.AMD_Radeon_pixmap)
    #         if self.rali_checkBox.isChecked():
    #             self.MIVisionX_logo.setPixmap(self.rali_pixmap)
    #             self.graph_imageLabel.setPixmap(self.graph_image_pixmap)
    #         else:
    #             self.MIVisionX_logo.setPixmap(self.MIVisionX_pixmap)
    #         self.EPYC_logo.setPixmap(self.EPYC_pixmap)
    #         self.totalCurve.setData(x=self.x, y=self.y, pen=self.pen)

    def closeEvent(self, event):
        self.receiver_thread.quit()
        self.close()
        exit(0)
