#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import sys


def frame_generator(filename):
    videoCapture = cv2.VideoCapture(filename)

    while True:
        success, frame = videoCapture.read()
        if not success:
            break
        yield frame


if __name__ == '__main__':
    for i, frame in enumerate(frame_generator(sys.argv[1])):
        cv2.imwrite('image/{0:04}.jpg'.format(i), frame)