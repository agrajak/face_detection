# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'lastproj.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 700)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.rand_img = QtWidgets.QPushButton(self.centralwidget)
        self.rand_img.setGeometry(QtCore.QRect(740, 590, 120, 45))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.rand_img.setFont(font)
        self.rand_img.setObjectName("rand_img")
        self.detect_btn = QtWidgets.QPushButton(self.centralwidget)
        self.detect_btn.setGeometry(QtCore.QRect(940, 590, 120, 45))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.detect_btn.setFont(font)
        self.detect_btn.setObjectName("detect_btn")
        self.val_acc = QtWidgets.QLabel(self.centralwidget)
        self.val_acc.setGeometry(QtCore.QRect(140, 500, 561, 101))
        font = QtGui.QFont()
        font.setFamily("HY헤드라인M")
        font.setPointSize(13)
        self.val_acc.setFont(font)
        self.val_acc.setTextFormat(QtCore.Qt.PlainText)
        self.val_acc.setAlignment(QtCore.Qt.AlignCenter)
        self.val_acc.setObjectName("val_acc")
        self.select_img = QtWidgets.QPushButton(self.centralwidget)
        self.select_img.setGeometry(QtCore.QRect(740, 510, 120, 45))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.select_img.setFont(font)
        self.select_img.setObjectName("select_img")
        self.before = QtWidgets.QLabel(self.centralwidget)
        self.before.setGeometry(QtCore.QRect(140, 120, 400, 300))
        self.before.setObjectName("before")
        self.select_video = QtWidgets.QPushButton(self.centralwidget)
        self.select_video.setGeometry(QtCore.QRect(940, 510, 120, 45))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.select_video.setFont(font)
        self.select_video.setObjectName("select_video")
        self.next_img = QtWidgets.QPushButton(self.centralwidget)
        self.next_img.setGeometry(QtCore.QRect(540, 440, 150, 50))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.next_img.setFont(font)
        self.next_img.setObjectName("next_img")
        self.after = QtWidgets.QLabel(self.centralwidget)
        self.after.setGeometry(QtCore.QRect(660, 120, 400, 300))
        self.after.setObjectName("after")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(540, 50, 150, 50))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.next_img.clicked['bool'].connect(MainWindow.show_img)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "FINALPROJECT 8조"))
        self.rand_img.setText(_translate("MainWindow", "랜덤 선택"))
        self.detect_btn.setText(_translate("MainWindow", "분류 시작"))
        self.val_acc.setText(_translate("MainWindow", "ACCURACY"))
        self.select_img.setText(_translate("MainWindow", "사진 선택하기"))
        self.select_video.setText(_translate("MainWindow", "영상 선택하기"))
        self.next_img.setText(_translate("MainWindow", "다음 사진"))
        self.pushButton.setText(_translate("MainWindow", "WebCam"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
