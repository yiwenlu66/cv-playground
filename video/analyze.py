# -*- coding: utf-8 -*-
# Author: Song Shihong

import cv2

videoCapture = cv2.VideoCapture('1.mov')

#获得码率及尺寸
fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

#读帧
success, frame = videoCapture.read()

c = 0

while success :
    cv2.waitKey(1000/int(fps)) #延迟
    cv2.imwrite('image/' + str(c) + '.jpg',frame) #写视频帧
    success, frame = videoCapture.read() #获取下一帧
    if c > 100:
        break
    c += 1
