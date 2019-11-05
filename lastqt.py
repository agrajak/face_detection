# -*- coding: utf-8 -*-
from lastpj import Ui_MainWindow
from datetime import datetime
import sys, cv2, numpy, time
import os.path as p
import os
import PyQt5.QtGui as QtGui 
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import random
from detector import Detector
# from PyQt5 import uic
#
# CUI = 'C:/Users/YCSEO/PycharmProjects/last2/lastproj.ui'


class qt__(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        # self.initUI()
        self.setupUi(self)
        print('waiting weights..')
        self.detector = Detector()
        print('loading weights done!')
        
        self.rand_img.clicked.connect(self.load_img)
        self.select_img.clicked.connect(self.select)
        self.select_video.clicked.connect(self.sel_video)
        self.detect_btn.clicked.connect(self.detected)
        self.pushButton.clicked.connect(self.webcam)
        
        self.it = None
        self.filepath = None
        self.list_img = None
        self.segimg = None
        self.img = None
    def webcam(self):
        self.sel_video(webcam=True)
        pass
    def detected(self):
        if self.segimg is None:
            return
        img = self.detector.resize(self.segimg)

        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
        boxes = self.detector.detect_face(img)
        before = (datetime.now().microsecond)
        ans = self.detector.get_name_from_file_path(self.filepath)

        txt = '정답: %s\n' % ans

        if len(boxes) == 0:
            print('no boxes!')
            txt = txt + '얼굴을 찾을 수 없습니다!\n'
        
        for (l, t, r, b) in boxes:
            
            cv2.rectangle(img, (l, t), (r, b),
                (0, 255, 0), 2)
            
            cropped_img = img[t:b,l:r]
            print('size of cropped img : ', cropped_img.shape)
            try:
                resized_img = cv2.resize(cropped_img, dsize=(90,90), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                print('error occured: ', str(e))
                return

            (predicted, prob_pred) = self.detector.predict(resized_img)
            
            text = "%s: %.2f" % (predicted, prob_pred*100)
            txt = txt + '예측: '+ text + '\n'
            text_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            y = t
            cv2.rectangle(img, (l,y-text_size[1]),(l+text_size[0], y+base_line), (0,255,0), -1)
            cv2.putText(img, text, (l, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        td = int((datetime.now().microsecond) - before)
        txt = txt + ('%.2lf초 소요\n' % (td/1000))
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
        self.val_acc.setText(txt)

        print ((datetime.now().microsecond) - before, 'ms ')

        qimg = QImage(img, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.after.setScaledContents(True)
        self.after.setPixmap(pix)

    def load_img(self):
        try:
            filepath = QFileDialog.getOpenFileNames(self)
            print(len(filepath[0]))

            self.list_img = filepath[0]

            random.shuffle(self.list_img)
            self.it = iter(self.list_img)
        except:
            pass

    def show_img(self):
        # 다음 사진
        if self.it is not None:
            path = next(self.it)
            self.segimg = cv2.imread(path, cv2.IMREAD_COLOR)
            self.filepath = path
            img = self.segimg

            cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)

            qimg = QImage(img, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)

            self.before.setScaledContents(True)
            self.before.setPixmap(pix)

    def select(self):
        try:
            filepath = QFileDialog.getOpenFileName(self)
            
            print(filepath[0])

            self.segimg = cv2.imread(filepath[0], 1)
            self.filepath = filepath[0]
            
            img = self.segimg
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
            qimg = QImage(img, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.before.setScaledContents(True)
            self.before.setPixmap(pix)
        except:
            pass

    def sel_video(self, webcam=False):
        try: 
            if webcam is False:
                filepath = QFileDialog.getOpenFileName(self)
                cap = cv2.VideoCapture(filepath[0])
            else:
                cap = cv2.VideoCapture(0)

            if cap is None:
                print("Can't find Video")
                exit(0)
            cnt = 0
            while(cap.isOpened()):
                cnt += 1
                ret, frame = cap.read()
                if cnt % 12 != 0:
                    continue
                print('frame shape is', frame.shape)
                img = self.detector.resize(cv2.cvtColor(frame, 1))

                boxes = self.detector.detect_face(img)
                if len(boxes) == 0:
                    print('no boxes!')
                
                for (l, t, r, b) in boxes:
                    print('l, t, r, b', l, t, r, b)
                    
                    cv2.rectangle(img, (l, t), (r, b),
                        (0, 255, 0), 2)
                    
                    cropped_img = img[t:b,l:r]
                    print('size of cropped img : ', cropped_img.shape)
                    try:
                        resized_img = cv2.resize(cropped_img, dsize=(90,90), interpolation=cv2.INTER_LINEAR)
                    except Exception as e:
                        print('error occured: ', str(e))
                        continue

                    (predicted, prob_pred) = self.detector.predict(resized_img)

                    text = "%s: %.2f" % (predicted, prob_pred*100)
                    text_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    y = t
                    cv2.rectangle(img, (l,y-text_size[1]),(l+text_size[0], y+base_line), (0,255,0), -1)
                    cv2.putText(img, text, (l, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                cv2.imshow('video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
        except:
            pass


app = QApplication([])
proj = qt__()
proj.show()
app.exec_()
