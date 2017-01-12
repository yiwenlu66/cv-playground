#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from tensorpack.dataflow.base import DataFlow
import glob
import os
import numpy as np
import cv2

IMG_W, IMG_H = 256, 256

class IdCard(DataFlow):

    def __init__(self, name, data_dir):
        self.data_dir = data_dir
        assert os.path.isdir(self.data_dir)

        assert name in ['train', 'val']
        self._load(name)

    def _load(self, name):
        raw_glob = os.path.join(self.data_dir, 'idcard', name, '*_raw.jpg')
        raw_files = sorted(glob.glob(raw_glob))
        gt_glob = os.path.join(self.data_dir, 'idcard', name, '*_gt.jpg')
        gt_files = sorted(glob.glob(gt_glob))
        assert len(raw_files) == len(gt_files)

        self.data = np.zeros((len(raw_files), IMG_H, IMG_W, 3), dtype='uint8')
        self.labels = np.zeros((len(raw_files), IMG_H, IMG_W), dtype='uint8')

        for idx, raw_file in enumerate(raw_files):

            gt_file = gt_files[idx]
            raw_name, gt_name = tuple(map(lambda path: os.path.basename(path).split('_')[0], (raw_file, gt_file)))
            assert raw_name == gt_name, 'name mismatch: {} != {}'.format(raw_name, gt_name)

            im = cv2.imread(raw_file, cv2.IMREAD_COLOR)
            assert im is not None
            assert im.shape[:2] == (IMG_H, IMG_W), "bad shape of {}: {} != {}".format(raw_name, im.shape[:2], (IMG_H, IMG_W))

            gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
            assert gt is not None
            assert gt.shape[:2] == (IMG_H, IMG_W), "bad shape of {}: {} != {}".format(gt_name, gt.shape[:2], (IMG_H, IMG_W))
            gt = np.vectorize(lambda x: 0 if x < 128 else 1)(gt)


            self.data[idx], self.labels[idx] = im, gt

    def size(self):
        return self.data.shape[0]

    def get_data(self):
        idxs = np.arange(self.data.shape[0])
        for k in idxs:
            yield [self.data[k], self.labels[k]]


if __name__ == '__main__':
    import sys
    a = IdCard('val', data_dir=sys.argv[1])
    for k in a.get_data():
        cv2.imshow("haha", k[0])
        cv2.waitKey(1000)
        cv2.imshow("haha", k[1]*255)
        cv2.waitKey(1000)