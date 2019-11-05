
# opencv-dnn (caffemodel 불러오기)코드는 https://github.com/Team-Neighborhood/awesome-face-detection/ 에서 참조하였습니다.


from __future__ import print_function
import numpy as np
import cv2
import argparse
import os
import os.path as p
import random
from model import vgg16

class Detector:
  def __init__(self):
    self.img_list = list()
    self.vggnet = vgg16()
    self.vggnet.build_model()    

    self.ans_n = 0
    self.wrong_n = 0
    self.no_box_n = 0
    self.verbose = True
    self.showImage = True
    self.waitKey = True
    self.isRandom = True
    
    self.net = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt', './models/res10_300x300_ssd_iter_140000.caffemodel')

  def load_images(self, input_dir):
    if p.isdir(input_dir) == False:
      print(input_dir, 'is not directory!')
      exit(1)

    for f in os.listdir(input_dir):
      if p.isfile(p.join(input_dir, f)):
        self.img_list.append(p.join(input_dir, f))

  def get_name_from_file_path(self, path):
    dict_names = {'TZUYU': 1, 'LEEKYUSANG': 2, 'IRENE':3, 'HWANGHEEJAE':4, 'mojihwan':5, 'BEN':6, 'sujin':7, 'IU':8, 'JAEIK': 9, 'SANA':10, 'LEEJEONGWOO': 11, 'CHAEUNWOO': 12, 'yj':13, 'dahyun':14, 'kdk':15, 'obama':16}

    name = path.split('\\')[-1].split('.')[0].split('_')[5]
    if name in dict_names:
      return name
    return None

  def predict(self, image):
    names = ['TZUYU', 'LEEKYUSANG', 'IRENE', 'HWANGHEEJAE', 'mojihwan', 'BEN', 'sujin', 'IU', 'JAEIK', 'SANA', 'LEEJEONGWOO', 'CHAEUNWOO', 'yj', 'dahyun', 'kdk', 'obama']

    data = list()
    data.append(image)
    input_data = np.asarray(data, dtype=np.float32)

    value = self.vggnet.predict(input_data).flatten()
    best = np.argmax(value)
    return (names[best], value[best])

  def detect_face(self, img):
    boxes = list()
    # dnn
    list_time = []
    for idx in range(10):
      start = cv2.getTickCount()
      (h, w) = img.shape[:2]

      blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
          (300, 300), (104.0, 177.0, 123.0))
      self.net.setInput(blob)
      detections = self.net.forward()

    for i in range(0, detections.shape[2]):
      confidence = detections[0, 0, i, 2]
      if confidence < 0.5:
        continue
      box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

      (l, t, r, b) = box.astype("int") # l t r b
      if l < 0:
        l = 0
      if t < 0:
        t = 0
      if r < 0:
        r = 0
      if b < 0:
        b = 0
      if l==r or t==b: 
        continue
      boxes.append((l, t, r, b))
    return boxes

  def resize(self, img, size=700):
    (h, w) = img.shape[:2]
    print('before resizing...', img.shape)
    if w > size:
      img = cv2.resize(img, dsize=(size, int(size*h/w)), interpolation=cv2.INTER_LINEAR)

    (h2, w2) = img.shape[:2]

    if h2 > size:
      img = cv2.resize(img, dsize=(int(size*w2/h2), size), interpolation=cv2.INTER_LINEAR)

    print('after resizing...', img.shape)

    return img

  def bench(self):
    for n in range(len(self.img_list)):
      if self.isRandom:
        image_path = self.img_list[int(random.random()*len(self.img_list))]
      else:
        image_path = self.img_list[n]

      bgr_img = self.resize(cv2.imread(image_path))
      hasAnswer = False
      boxes = self.detect_face(bgr_img)
      ans = self.get_name_from_file_path(image_path)

      green = (0,255,0)
      red = (255,0,0)
      blue = (0,0,255)

      if ans is None:
        print(image_path, '에서 이름을 찾을 수 없습니다.')
        continue

      if len(boxes) == 0:
        self.no_box_n += 1
        continue

      for (l, t, r, b) in boxes:
        print('l, t, r, b', l, t, r, b)
        
        cv2.rectangle(bgr_img, (l, t), (r, b),
            green, 2)

        (x, y, w, h) = image_path.split('\\')[-1].split('.')[0].split('_')[1:5]
          
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        
        cv2.rectangle(bgr_img, (x, y), (x+w, y+h),
            blue, 1)

        cropped_img = bgr_img[t:b,l:r]
        print('size of cropped img : ', cropped_img.shape)
        try:
          resized_img = cv2.resize(cropped_img, dsize=(90,90), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
          print('error occured: ', str(e))
          continue

        (predicted, prob_pred) = self.predict(resized_img)
        
        color = red
        ans_str = 'X'
        print('predicted, ', predicted, 'ans', ans)
        if predicted == ans:
          hasAnswer = True
          ans_str = 'O'
          color = green
        
        text = "%s: %.2f (%s)" % (predicted, prob_pred*100, ans_str)
        text_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y = t
        cv2.rectangle(bgr_img, (l,y-text_size[1]),(l+text_size[0], y+base_line), (0,255,0), -1)
        cv2.putText(bgr_img, text, (l, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        
        if self.showImage:
          cv2.imshow('show', bgr_img)
          if self.waitKey:
            cv2.waitKey()

      if hasAnswer:
        self.ans_n += 1
      else:
        self.wrong_n += 1
        
      if self.verbose:
        if n%20 == 0:
          print('현재 진행', n, '/', len(self.img_list))
          total = self.ans_n + self.no_box_n + self.wrong_n

          print('박스가 없을 확률: %.2lf' % (self.no_box_n/total*100))
          print('정확도: %.2lf' % (self.ans_n/(self.ans_n + self.wrong_n)*100))

    if self.verbose:
      print('===최종결과===')
      total = self.ans_n + self.no_box_n + self.wrong_n

      print('박스가 없을 확률: %.2lf' % (self.no_box_n/total*100))
      print('정확도: %.2lf' % (self.ans_n/(self.ans_n + self.wrong_n)*100))

