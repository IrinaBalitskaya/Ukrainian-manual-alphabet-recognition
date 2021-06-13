import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from option_ui import Ui_MainWindow
from webcamRecognition import WebcamRecognition
from imageRecognition import ImageRecognition
from videoRecognition import VideoRecognition


class Option(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.webcam_btn = self.radioButton
        self.video_btn = self.radioButton_3
        self.image_btn = self.radioButton_2
        self.pushButton.clicked.connect(self.openWindow)

    def openWindow(self):
        if self.webcam_btn.isChecked():
            self.window = WebcamRecognition()
        elif self.video_btn.isChecked():
            self.window = VideoRecognition()
        elif self.image_btn.isChecked():
            self.window = ImageRecognition()

        self.window.show()


if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = Option()
    Root.show()
    sys.exit(App.exec())

