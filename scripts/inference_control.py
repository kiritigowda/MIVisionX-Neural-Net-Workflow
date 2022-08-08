import os
import enum
#from PyQt4 import QtGui, uic
from PyQt5 import QtWidgets, uic
from inference_viewer import *
from train_viewer import *

class Mode(enum.Enum):
    TRAINING = 0,
    INFERENCE = 1

class InferenceControl(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(InferenceControl, self).__init__(parent)
        self.runningState = False
        self.mode = None
        self.initUI()

    def initUI(self):
        uic.loadUi("inference_control.ui", self)
        #self.setStyleSheet("background-color: white")
        self.tabWidget.setCurrentIndex(0)
        self.ti_pushButton.clicked.connect(self.confirmMode)
        self.close_pushButton.clicked.connect(self.closeEvent)
        self.run_pushButton.clicked.connect(self.runConfig)
        self.run_pushButton.setEnabled(False)
        self.close_pushButton.setStyleSheet("color: white; background-color: darkRed")

        #inference
        self.upload_comboBox.activated.connect(self.fromFile)
        self.file_pushButton.clicked.connect(self.browseFile)
        self.output_pushButton.clicked.connect(self.browseOutput)
        self.label_pushButton.clicked.connect(self.browseLabel)
        self.image_pushButton.clicked.connect(self.browseImage)
        self.val_pushButton.clicked.connect(self.browseVal)
        self.hier_pushButton.clicked.connect(self.browseHier)
        self.file_lineEdit.textChanged.connect(self.checkInput)
        self.name_lineEdit.textChanged.connect(self.checkInput)
        self.idims_lineEdit.textChanged.connect(self.checkInput)
        self.odims_lineEdit.textChanged.connect(self.checkInput)
        self.output_lineEdit.textChanged.connect(self.checkInput)
        self.label_lineEdit.textChanged.connect(self.checkInput)
        self.image_lineEdit.textChanged.connect(self.checkInput)
        self.image_lineEdit.textChanged.connect(self.checkInput)
        self.file_lineEdit.setPlaceholderText("File Directory [required]")
        self.name_lineEdit.setPlaceholderText("Model Name [required]")
        self.idims_lineEdit.setPlaceholderText("c,h,w [required]")
        self.odims_lineEdit.setPlaceholderText("c,h,w [required]")
        self.padd_lineEdit.setPlaceholderText("r,g,b [optional]")
        self.pmul_lineEdit.setPlaceholderText("r,g,b [optional]")
        self.output_lineEdit.setPlaceholderText("Output Directory [required]")
        self.label_lineEdit.setPlaceholderText("Label File [required]")
        self.image_lineEdit.setPlaceholderText("Image Folder [required]")
        self.val_lineEdit.setPlaceholderText("[optional]")
        self.hier_lineEdit.setPlaceholderText("[optional]")
        self.mode_comboBox.setEnabled(True)
        self.gui_checkBox.setChecked(True)
        self.readSetupFile()

        # training
        self.tidims_lineEdit.setPlaceholderText("n,c,h,w [required]")
        self.dpath_lineEdit.setPlaceholderText("Data Directory [required]")
        self.opath_lineEdit.setPlaceholderText("Output Directory [required]")
        self.numgpu_lineEdit.setPlaceholderText("GPU(s)")
        self.epoch_lineEdit.setPlaceholderText("Epoch(s)")
        self.tgpu_radioButton.setChecked(1)
        self.rcpu_radioButton.setChecked(1)
        self.model_comboBox.currentIndexChanged.connect(self.checkTInput)
        self.dtype_comboBox.currentIndexChanged.connect(self.checkTInput)
        self.tidims_lineEdit.textChanged.connect(self.checkTInput)
        self.opath_lineEdit.textChanged.connect(self.checkTInput)
        self.dpath_lineEdit.textChanged.connect(self.checkTInput)
        self.numgpu_lineEdit.textChanged.connect(self.checkTInput)
        self.epoch_lineEdit.textChanged.connect(self.checkTInput)
        self.dpath_pushButton.clicked.connect(self.browseDPath)
        self.opath_pushButton.clicked.connect(self.browseOPath)
        self.tgui_checkBox.setChecked(True)
    
    def confirmMode(self):
        self.tabWidget.setEnabled(True)
        self.ti_comboBox.setEnabled(False)
        self.ti_pushButton.setEnabled(False)

        if self.ti_comboBox.currentIndex() == 0:
            self.mode = Mode.TRAINING
            self.tabWidget.setCurrentIndex(0)
            self.tabWidget.setTabEnabled(0, True)
            self.tabWidget.setTabEnabled(1, False)
        elif self.ti_comboBox.currentIndex() == 1:
            self.mode = Mode.INFERENCE
            self.tabWidget.setCurrentIndex(1)
            self.tabWidget.setTabEnabled(1, True)
            self.tabWidget.setTabEnabled(0, False)



    def browseFile(self):
        if self.format_comboBox.currentText() == 'nnef':
            self.file_lineEdit.setText(QtWidgets.QFileDialog.getExistingDirectory(self, 'Open Folder', './'))    
        else:
            self.file_lineEdit.setText(QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', './')[0])

    def browseDPath(self):
        self.dpath_lineEdit.setText(QtWidgets.QFileDialog.getExistingDirectory(self, 'Open Folder', './'))

    def browseOPath(self):
        self.opath_lineEdit.setText(QtWidgets.QFileDialog.getExistingDirectory(self, 'Open Folder', './'))

    def browseOutput(self):
        self.output_lineEdit.setText(QtWidgets.QFileDialog.getExistingDirectory(self, 'Open Folder', './')[0])

    def browseLabel(self):
        self.label_lineEdit.setText(QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', './', '*.txt')[0])

    def browseImage(self):
        self.image_lineEdit.setText(QtWidgets.QFileDialog.getExistingDirectory(self, 'Open Folder', './'))

    def browseVal(self):
        self.val_lineEdit.setText(QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', './', '*.txt')[0])

    def browseHier(self):
        self.hier_lineEdit.setText(QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', './', '*.csv')[0])

    def readSetupFile(self):
        setupDir = '~/.mivisionx-neuralnet-workflow'
        analyzerDir = os.path.expanduser(setupDir)
        if os.path.isfile(analyzerDir + "/setupFile.txt"):
            for line in open(analyzerDir + "/setupFile.txt", "r"):
                if len(line) > 1:
                    token = line.split(';')
                    if len(token) > 1:
                        modelName = token[1]
                        self.upload_comboBox.addItem(modelName)
            
    def fromFile(self):
        if self.upload_comboBox.currentIndex() == 0:
            self.name_lineEdit.setEnabled(True)
            self.file_lineEdit.setEnabled(True)
            self.batch_comboBox.setEnabled(True)
            self.mode_comboBox.setCurrentIndex(0)
            self.idims_lineEdit.setEnabled(True)
            self.odims_lineEdit.setEnabled(True)
            self.label_lineEdit.setEnabled(True)
            self.output_lineEdit.setEnabled(True)
            self.image_lineEdit.setEnabled(True)
            self.val_lineEdit.setEnabled(True)
            self.hier_lineEdit.setEnabled(True)
            self.padd_lineEdit.setEnabled(True)
            self.pmul_lineEdit.setEnabled(True)
            self.gui_checkBox.setEnabled(True)
            self.fp16_checkBox.setEnabled(True)
            self.replace_checkBox.setEnabled(True)
            self.verbose_checkBox.setEnabled(True)
            self.loop_checkBox.setEnabled(True)
            self.file_pushButton.setEnabled(True)
            self.format_comboBox.setEnabled(True)
            self.output_pushButton.setEnabled(True)
            self.label_pushButton.setEnabled(True)
            self.image_pushButton.setEnabled(True)
            self.val_pushButton.setEnabled(True)
            self.hier_pushButton.setEnabled(True)
            self.format_comboBox.setCurrentIndex(0)
            self.name_lineEdit.clear()
            self.file_lineEdit.clear()
            self.idims_lineEdit.clear()
            self.odims_lineEdit.clear()
            self.label_lineEdit.clear()
            self.output_lineEdit.clear()
            self.image_lineEdit.clear()
            self.val_lineEdit.clear()
            self.hier_lineEdit.clear()
            self.padd_lineEdit.clear()
            self.pmul_lineEdit.clear()
            self.gui_checkBox.setChecked(True)
            self.fp16_checkBox.setChecked(False)
            self.replace_checkBox.setChecked(False)
            self.verbose_checkBox.setChecked(False)
            self.loop_checkBox.setChecked(True)
        else:
            modelName = self.upload_comboBox.currentText()
            setupDir = '~/.mivisionx-neuralnet-workflow'
            analyzerDir = os.path.expanduser(setupDir)
            for line in open(analyzerDir + "/setupFile.txt", "r"):
                tokens = line.split(';')
                if len(tokens) > 1:
                    name = tokens[1]
                    if modelName == name:
                        if tokens[0] == 'caffe':
                            format = 0
                        elif tokens[0] == 'onnx':
                            format = 1
                        else:
                            format = 2
                        self.format_comboBox.setCurrentIndex(format)
                        self.name_lineEdit.setText(tokens[1])
                        self.file_lineEdit.setText(tokens[2])
                        batch_index = 0 if tokens[3] == '64' else 1
                        self.batch_comboBox.setCurrentIndex(batch_index)
                        self.idims_lineEdit.setText(tokens[4])
                        self.odims_lineEdit.setText(tokens[5])
                        self.label_lineEdit.setText(tokens[6])
                        self.output_lineEdit.setText(tokens[7])
                        self.image_lineEdit.setText(tokens[8])
                        self.val_lineEdit.setText(tokens[9])
                        self.hier_lineEdit.setText(tokens[10])
                        self.padd_lineEdit.setText(tokens[11])
                        self.pmul_lineEdit.setText(tokens[12])
                        self.fp16_checkBox.setChecked(True) if tokens[13] == 'yes\n' or tokens[13] == 'yes' else self.fp16_checkBox.setChecked(False)
                        self.replace_checkBox.setChecked(True) if tokens[14] == 'yes\n' or tokens[14] == 'yes' else self.replace_checkBox.setChecked(False)
                        self.verbose_checkBox.setChecked(True) if tokens[15] == 'yes\n' or tokens[15] == 'yes' else self.verbose_checkBox.setChecked(False)                        
                        self.loop_checkBox.setChecked(True) if tokens[16] == 'yes\n' or tokens[16] == 'yes' else self.replace_checkBox.setChecked(False)
                        self.name_lineEdit.setEnabled(False)
                        self.file_lineEdit.setEnabled(False)
                        self.batch_comboBox.setEnabled(False)
                        self.idims_lineEdit.setEnabled(False)
                        self.odims_lineEdit.setEnabled(False)
                        self.label_lineEdit.setEnabled(False)
                        self.output_lineEdit.setEnabled(False)
                        self.image_lineEdit.setEnabled(False)
                        self.val_lineEdit.setEnabled(False)
                        self.hier_lineEdit.setEnabled(False)
                        self.padd_lineEdit.setEnabled(False)
                        self.pmul_lineEdit.setEnabled(False)
                        self.fp16_checkBox.setEnabled(False)
                        self.output_pushButton.setEnabled(False)
                        self.label_pushButton.setEnabled(False)
                        self.image_pushButton.setEnabled(False)
                        self.val_pushButton.setEnabled(False)
                        self.hier_pushButton.setEnabled(False)
                        self.verbose_checkBox.setEnabled(False)
                        self.file_pushButton.setEnabled(False)
                        self.format_comboBox.setEnabled(False)

    def checkInput(self):
        if not (self.file_lineEdit.text() == '') and not (self.name_lineEdit.text() == '') \
            and not (self.idims_lineEdit.text() == '')  and not (self.odims_lineEdit.text() == '') \
            and not (self.output_lineEdit.text() == '') and not (self.label_lineEdit.text() == '') \
            and not (self.image_lineEdit.text() == ''):
                self.run_pushButton.setEnabled(True)
                self.run_pushButton.setStyleSheet("background-color: lightgreen")
        else:
            self.run_pushButton.setEnabled(False)
            self.run_pushButton.setStyleSheet("background-color: 0")

    def checkTInput(self):
        if not (self.model_comboBox.currentIndex() == 0) and not (self.dtype_comboBox.currentIndex() == 0) \
            and not (self.dpath_lineEdit.text() == '') and not (self.opath_lineEdit.text() == '') \
            and not (self.tidims_lineEdit.text() == '') and not (self.numgpu_lineEdit.text() == '') \
            and not (self.epoch_lineEdit.text() == ''):
                self.run_pushButton.setEnabled(True)
                self.run_pushButton.setStyleSheet("background-color: lightgreen")
        else:
            self.run_pushButton.setEnabled(False)
            self.run_pushButton.setStyleSheet("background-color: 0")
            
    def runTraining(self):
        model = (str)(self.model_comboBox.currentText())
        datapath = (str)(self.dpath_lineEdit.text())
        dataset_train =  datapath + '/train'
        if not os.path.exists(dataset_train):
            msg = QtWidgets.QMessageBox.critical(self, "Error", "Invalid dataset, requires train folder")
            return
        dataset_val = datapath + '/val'
        if not os.path.exists(dataset_val):
            msg = QtWidgets.QMessageBox.critical(self, "Error", "Invalid dataset, requires val folder")
            return

        PATH = (str)(self.opath_lineEdit.text())
        training_device = self.tgpu_radioButton.isChecked()
        num_gpu = (int)(self.numgpu_lineEdit.text())
        batch_size = (int)(self.batch_comboBox.currentText())
        epochs = (int)(self.epoch_lineEdit.text())
        rocal_cpu = self.rcpu_radioButton.isChecked()
        input_dims = (str)('%s' % (self.tidims_lineEdit.text()))
        num_thread = 1
        gui = self.tgui_checkBox.isChecked()

        self.runningState = True
        self.close()
        trainer = TrainViewer(model, datapath, PATH, training_device, num_gpu, batch_size, epochs, rocal_cpu, input_dims, num_thread, gui, self)
        # trainer.showMaximized()
        # trainer.show()


    def runInference(self):
        model_format = (str)(self.format_comboBox.currentText())
        model_name = (str)(self.name_lineEdit.text())
        model_location = (str)(self.file_lineEdit.text())
        batch_size = (str)(self.batch_comboBox.currentText())
        rocal_mode = self.mode_comboBox.currentIndex() + 1 
        input_dims = (str)('%s' % (self.idims_lineEdit.text()))
        output_dims = (str)('%s' % (self.odims_lineEdit.text()))
        label = (str)(self.label_lineEdit.text())
        output_dir = (str)(self.output_lineEdit.text())
        image_dir = (str)(self.image_lineEdit.text())
        image_val = (str)(self.val_lineEdit.text())
        hierarchy = (str)(self.hier_lineEdit.text())
        if len(self.padd_lineEdit.text()) < 1:
            add = '[0,0,0]'
        else:
            add = (str)('[%s]' % (self.padd_lineEdit.text()))
        if len(self.pmul_lineEdit.text()) < 1:
            multiply = '[1,1,1]'
        else:
            multiply = (str)('[%s]' % (self.pmul_lineEdit.text()))
        gui = 'yes' if self.gui_checkBox.isChecked() else 'no'
        fp16 = 'yes' if self.fp16_checkBox.isChecked() else 'no'
        replace = 'yes' if self.replace_checkBox.isChecked() else 'no'
        verbose = 'yes' if self.verbose_checkBox.isChecked() else 'no'
        loop = 'yes' if self.loop_checkBox.isChecked() else 'no'
        container_logo = self.container_comboBox.currentIndex()
        fps_file = ''
        cpu_name = self.cpu_comboBox.currentText()
        gpu_name = self.gpu_comboBox.currentText()
        self.runningState = True

        self.close()

        viewer = InferenceViewer(model_name, model_format, image_dir, model_location, label, hierarchy, image_val, input_dims, output_dims, batch_size, output_dir, 
                                    add, multiply, verbose, fp16, replace, loop, rocal_mode, gui, container_logo, fps_file, cpu_name, gpu_name, self)
        if gui == 'yes':
            #viewer.show()
            viewer.showMaximized()
            

    def runConfig(self):
        if self.mode == Mode.TRAINING:
            self.runTraining()
        elif self.mode == Mode.INFERENCE:
            self.runInference()
            

    def closeEvent(self, event):
        if not self.runningState:
            exit(0)
