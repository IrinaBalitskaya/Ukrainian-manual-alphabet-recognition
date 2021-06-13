import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from webcam_ui import Ui_MainWindow
import cv2

from ImageClassification import imageClassification, imagePreparation, letterOutput

IMG_SIZE = 60


class WebcamRecognition(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.feed_label = self.label
        self.start_cancel_button = self.pushButton
        self.save_button = self.pushButton_2
        self.worker = Worker1()
        self.start_cancel_button.clicked.connect(self.StartCancelFeed)
        self.save_button.clicked.connect(self.saveTranslation)
        self.worker.image_update.connect(self.image_updateSlot)
        self.worker.TextUpdate.connect(self.TextBrowserUpdate)

    def image_updateSlot(self, Image):
        self.feed_label.setPixmap(QPixmap.fromImage(Image))

    def TextBrowserUpdate(self, letter):
        self.textBrowser.append(letter)

    def saveTranslation(self):
        text = self.textBrowser.toPlainText()
        with open('webcam_save.txt', 'w') as f:
            for txt in text:
                f.write(txt)
        print(f'Збережено в webcam_save.txt')

    def StartCancelFeed(self):
        if self.start_cancel_button.text() == 'Почати':
            self.worker.start()
            self.start_cancel_button.setText('Зупинити')
        elif self.start_cancel_button.text() == 'Зупинити':
            self.worker.stop()
            self.start_cancel_button.setText('Почати')


class Worker1(QThread):
    image_update = pyqtSignal(QImage)
    TextUpdate = pyqtSignal(str)

    def run(self):
        self.ThreadActive = True
        capture = cv2.VideoCapture(0)
        letter_list = []
        while self.ThreadActive:
            ret, img = capture.read()
            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.flip(img, 1)

                convertToQtFormat = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
                pic = convertToQtFormat.scaled(350, 350, Qt.KeepAspectRatio)
                self.image_update.emit(pic)

                img = imagePreparation(img, IMG_SIZE, False)
                predicted_letter, probability = imageClassification(img, IMG_SIZE)
                if probability > 0.9:
                    print(f"Predicted class: {predicted_letter} - {probability:.2f}%")
                letter_list = letterOutput(predicted_letter, letter_list, 2)
                if len(letter_list) == 60:
                    self.TextUpdate.emit(predicted_letter)
        capture.release()

    def stop(self):
        self.ThreadActive = False
        self.quit()


if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = WebcamRecognition()
    Root.show()
    sys.exit(App.exec())
