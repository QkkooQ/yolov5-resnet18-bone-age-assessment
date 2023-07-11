import sys
import typing

from PyQt5.QtWidgets import *
from PyQt5 import QtCore, uic
from PyQt5.QtWidgets import QWidget, QFileDialog
from scorer import load_model, detect_img

from PIL import Image
from PyQt5.QtGui import QPixmap,QImage

class MyWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.yolov5_model, self.cls_models = load_model()
        self.init_ui()

    def init_ui(self):
        self.ui = uic.loadUi("bone_test.ui")

        self.upload_btn = self.ui.upload
        self.detect_btn = self.ui.detect

        self.man_btn = self.ui.male
        self.man_btn.setChecked(True)
        self.woman_btn = self.ui.female

        self.img_label = self.ui.input
        self.result_label = self.ui.output

        self.upload_btn.clicked.connect(self.upload_img)
        self.detect_btn.clicked.connect(self.detect_img)

    def upload_img(self):
        self.img_path, _ = QFileDialog.getOpenFileName(self, "Open file", "c:\\", "Image files (*.jpg *.png *.jpeg)")

        im = Image.open(self.img_path)
        im = im.resize((271,431))
        im = im.convert("RGBA")
        data = im.tobytes("raw","RGBA")
        qim = QImage(data,im.size[0],im.size[1],QImage.Format_ARGB32)
        pixmap = QPixmap.fromImage(qim)
        self.img_label.setPixmap(pixmap)

    def detect_img(self):
        sex = "boy" if self.man_btn.isChecked() else "girl"
        export = detect_img(self.yolov5_model, self.cls_models, self.img_path, sex)
        self.result_label.setText(export)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MyWindow()
    w.ui.show()
    app.exec()
