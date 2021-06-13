import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from images_ui import Ui_MainWindow
from PyQt5.QtWidgets import QFileDialog
import glob
import os
from ImageClassification import imageClassification, imagePreparation


class ImageRecognition(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.backward_btn = self.pushButton
        self.forward_btn = self.pushButton_2
        self.save_button = self.pushButton_3
        self.folder_btn = self.pushButton_4
        self.image_size = 60
        self.image_list = []
        self.current_image = ''
        self.directory = os.getcwd()
        self.folder_btn.clicked.connect(self.selectFolder)
        self.save_button.clicked.connect(self.saveTranslation)
        self.forward_btn.clicked.connect(self.scrollForward)
        self.backward_btn.clicked.connect(self.scrollBackward)

    def selectFolder(self):
        self.textBrowser.clear()
        self.directory = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.image_list = glob.glob(f'{self.directory}/*.jpg')
        if self.image_list:
            self.image_updateSlot(self.image_list[0])
            self.current_image = self.image_list[0]
            self.updateLabel()
            self.textBrowserUpdate()
        return self.directory

    def image_updateSlot(self, Image):
        self.label.setPixmap(QPixmap(Image))
        self.label.setAlignment(Qt.AlignCenter)

    def textBrowserUpdate(self):
        for img in self.image_list:
            img = imagePreparation(img, self.image_size, True)
            letter, _ = imageClassification(img, self.image_size)
            self.textBrowser.append(letter)

    def scrollForward(self):
        if self.current_image != '':
            if self.image_list.index(self.current_image) < len(self.image_list) - 1:
                self.current_image = self.image_list[self.image_list.index(self.current_image) + 1]
                self.image_updateSlot(self.current_image)
            else:
                self.current_image = self.image_list[0]
                self.image_updateSlot(self.current_image)
        self.updateLabel()

    def scrollBackward(self):
        if self.current_image != '':
            if self.image_list.index(self.current_image) > 0:
                self.current_image = self.image_list[self.image_list.index(self.current_image) - 1]
                self.image_updateSlot(self.current_image)
            else:
                self.current_image = self.image_list[len(self.image_list) - 1]
                self.image_updateSlot(self.current_image)
        self.updateLabel()

    def updateLabel(self):
        self.label_2.setText(f'{self.image_list.index(self.current_image) + 1}/{len(self.image_list)}')

    def saveTranslation(self):
        text = self.textBrowser.toPlainText()
        print(text)
        with open('image_save.txt', 'w') as f:
            for txt in text:
                f.write(txt)


if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = ImageRecognition()
    Root.show()
    sys.exit(App.exec())
