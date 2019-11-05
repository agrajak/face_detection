from detector import Detector
import os
import os.path as p
import cv2 as cv

def get_video_type(filename):
  VIDEO_TYPE = {
    'avi': cv.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv.VideoWriter_fourcc(*'XVID'),
  }
  filename, ext = os.path.splitext(filename)
  if ext in VIDEO_TYPE:
    return  VIDEO_TYPE[ext]
  return VIDEO_TYPE['avi']

if __name__ == '__main__':
  Detect = Detector()

  video_list = []
  video_output_list = []
  input_dir = p.normpath(os.getcwd()+'/videos')
  if p.isdir(input_dir) == False:
    print(input_dir, 'is not directory!')
    exit(1)

  for f in os.listdir(input_dir):
    output_dir = p.normpath(os.getcwd()+'/output')
    if p.exists(output_dir) == False:
      os.mkdir(output_dir)
    _f = p.join(input_dir, f)
    if p.isfile(_f):
      video_list.append(_f)
      video_output_list.append(p.join(output_dir, f))

  print('비디오의 개수: ', len(video_list))

  for i in range(len(video_list)):
    videoPath = video_list[i]
    outputPath = video_output_list[i]
    print('opening videoPath: ', videoPath)
    print('opening outputPath: ', outputPath)
    cap = cv.VideoCapture(videoPath)

    cap.set(cv.CAP_PROP_POS_FRAMES, 15)

    out = None

    if cap is None:
      print("Can't find Video")
      exit(0)

    while(cap.isOpened()):
      ret, frame = cap.read()
      if ret is None:
        break
      print('frame shape is', frame.shape)
      img = Detect.resize(cv.cvtColor(frame, 1))

      boxes = Detect.detect_face(img)
      if len(boxes) == 0:
        print('no boxes!')
        
      for (l, t, r, b) in boxes:
        print('l, t, r, b', l, t, r, b)
        
        cv.rectangle(img, (l, t), (r, b),
            (0, 255, 0), 2)
        
        cropped_img = img[t:b,l:r]
        print('size of cropped img : ', cropped_img.shape)
        try:
          resized_img = cv.resize(cropped_img, dsize=(90,90), interpolation=cv.INTER_LINEAR)
        except Exception as e:
          print('error occured: ', str(e))
          continue

        (predicted, prob_pred) = Detect.predict(resized_img)

        text = "%s: %.2f" % (predicted, prob_pred*100)
        text_size, base_line = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
        y = t
        cv.rectangle(img, (l,y-text_size[1]),(l+text_size[0], y+base_line), (0,255,0), -1)
        cv.putText(img, text, (l, y),
            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

      if out is None:
        width = img.shape[1]
        height = img.shape[0]
        out = cv.VideoWriter(outputPath, get_video_type(videoPath), 25, (width, height))
      out.write(img)
        
    cap.release()
    out.release()