#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from tensorpack.dataflow.base import DataFlow
import glob
import os
import numpy as np
import cv2
from scipy.misc import imresize


IMG_H, IMG_W = 256, 256    # force scale to 256x256


def copyMakeBorderWithRandomNoise(src, top, bottom, left, right):
    result = np.zeros((top + src.shape[0] + bottom, left + src.shape[1] + right, src.shape[2]), dtype='uint8')
    padtop = np.random.randint(0, 256, size=(top, left + src.shape[1] + right, src.shape[2]), dtype='uint8')
    padbottom = np.random.randint(0, 256, size=(bottom, left + src.shape[1] + right, src.shape[2]), dtype='uint8')
    padleft = np.random.randint(0, 256, size=(src.shape[0], left, src.shape[2]), dtype='uint8')
    padright = np.random.randint(0, 256, size=(src.shape[0], right, src.shape[2]), dtype='uint8')
    result[:top, :, :] = padtop
    result[top:top + src.shape[0], :left, :] = padleft
    result[top:top + src.shape[0], left:left + src.shape[1], :] = src
    result[top:top + src.shape[0], left + src.shape[1]:, :] = padright
    result[top + src.shape[0]:, :, :] = padbottom
    return result


class PennFudanPed(DataFlow):

    def __init__(self, name, data_dir):
        self.data_dir = data_dir
        assert os.path.isdir(self.data_dir)

        assert name in ['train', 'val']
        self._load(name)

    def _load(self, name):
        # Use Penn for training, Fudan for evaluation
        wildcard = 'PennPed*.png' if name == 'train' else 'FudanPed*.png'
        raw_glob = os.path.join(self.data_dir, 'PennFudanPed', 'PNGImages', wildcard)
        raw_files = sorted(glob.glob(raw_glob))
        gt_glob = os.path.join(self.data_dir, 'PennFudanPed', 'PedMasks', wildcard)
        gt_files = sorted(glob.glob(gt_glob))
        assert len(raw_files) == len(gt_files)

        self.data = np.zeros((len(raw_files), IMG_H, IMG_W, 3), dtype='uint8')
        self.labels = np.zeros((len(raw_files), IMG_H, IMG_W), dtype='uint8')

        for idx, raw_file in enumerate(raw_files):
            gt_file = gt_files[idx]
            raw_name = os.path.basename(raw_file).split('.')[0]
            gt_name = os.path.basename(gt_file).split('_')[0]
            assert raw_name == gt_name, 'name mismatch: {} != {}'.format(raw_name, gt_name)

            im = cv2.imread(raw_file, cv2.IMREAD_COLOR)
            assert im is not None
            gt = cv2.imread(gt_file, cv2.IMREAD_COLOR)
            assert gt is not None
            assert gt.shape == im.shape, 'shape mismatch: {}, {} != {}'.format(gt_name, gt.shape, im.shape)

            k = min(1.0 * IMG_H / im.shape[0], 1.0 * IMG_W / im.shape[1])
            newshape = (min(int(k * im.shape[0]), IMG_H), min(int(k * im.shape[1]), IMG_W))
            im = imresize(im, newshape)
            tborder, bborder, lborder, rborder = 0, 0, 0, 0
            if im.shape[0] < IMG_H:
                tborder = (IMG_H - im.shape[0]) / 2
                bborder = IMG_H - im.shape[0] - tborder
            if im.shape[1] < IMG_W:
                lborder = (IMG_W - im.shape[1]) / 2
                rborder = IMG_W - im.shape[1] - lborder
            im = copyMakeBorderWithRandomNoise(im, tborder, bborder, lborder, rborder)

            gt = np.any(gt, axis=2).astype('uint8')
            gt = imresize(gt, newshape)
            BLACK = [0, 0, 0]
            gt = cv2.copyMakeBorder(gt, tborder, bborder, lborder, rborder, cv2.BORDER_CONSTANT, value=BLACK)

            self.data[idx], self.labels[idx] = im, gt

    def size(self):
        return self.data.shape[0]

    def get_data(self):
        idxs = np.arange(self.data.shape[0])
        for k in idxs:
            yield [self.data[k], self.labels[k]]


if __name__ == '__main__':
    import sys
    a = PennFudanPed('val', data_dir=sys.argv[1])
    for k in a.get_data():
        cv2.imshow("haha", k[0])
        cv2.waitKey(1000)
        cv2.imshow("haha", k[1]*255)
        cv2.waitKey(1000)