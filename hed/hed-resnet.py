#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np
import tensorflow as tf
import argparse
from six.moves import zip
import os, sys

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from idcard_dataset import IdCard

from tensorflow.contrib.layers import variance_scaling_initializer

DATA_DIR = '../data/'

N = 5


class Model(ModelDesc):

    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, None, None, 3], 'image'),
                InputVar(tf.int32, [None, None, None], 'heatmap')]

    def _build_graph(self, input_vars):
        image, heatmap = input_vars
        image = image - tf.constant([104, 116, 122], dtype='float32')
        heatmap = tf.expand_dims(heatmap, 3, name='heatmap4d')

        def branch(name, l, up):
            with tf.variable_scope(name) as scope:
                l = Conv2D('convfc', l, 1, kernel_shape=1, nl=tf.identity,
                           use_bias=True,
                           W_init=tf.constant_initializer(),
                           b_init=tf.constant_initializer())
                while up != 1:
                    l = BilinearUpSample('upsample{}'.format(up), l, 2)
                    up = up / 2
                return l


        def residual(name, l, increase_dim=False, first=False):
            shape = l.get_shape().as_list()
            in_channel = shape[3]

            if increase_dim:
                out_channel = in_channel * 2
                stride1 = 2
            else:
                out_channel = in_channel
                stride1 = 1

            with tf.variable_scope(name) as scope:
                b1 = l if first else BNReLU(l)
                c1 = Conv2D('conv1', b1, out_channel, stride=stride1, nl=BNReLU)
                c2 = Conv2D('conv2', c1, out_channel)
                if increase_dim:
                    l = AvgPooling('pool', l, 2)
                    l = tf.pad(l, [[0, 0], [0, 0], [0, 0], [in_channel // 2, in_channel // 2]])

                l = c2 + l
                return l


        with argscope(Conv2D, kernel_shape=3, nl=tf.identity, use_bias=False,
            W_init=variance_scaling_initializer(mode='FAN_OUT')):

            l = Conv2D('conv0', image, 16, nl=BNReLU)
            l = residual('res1.0', l, first=True)
            for k in range(1, N):
                l = residual('res1.{}'.format(k), l)
            b1 = branch('branch1', l, 1)
            # 32,c=16

            l = residual('res2.0', l, increase_dim=True)
            for k in range(1, N):
                l = residual('res2.{}'.format(k), l)
            b2 = branch('branch2', l, 2)
            # 16,c=32

            l = residual('res3.0', l, increase_dim=True)
            for k in range(1, N):
                l = residual('res3.' + str(k), l)
            l = BNReLU('bnlast', l)
            b3 = branch('branch3', l, 4)
            # 8,c=64
            l = GlobalAvgPooling('gap', l)

        final_map = Conv2D('convfcweight',
                           tf.concat_v2([b1, b2, b3], 3), 1, 1,
                           W_init=tf.constant_initializer(0.2),
                           use_bias=False, nl=tf.identity)
        costs = []
        for idx, b in enumerate([b1, b2, b3, final_map]):
            output = tf.nn.sigmoid(b, name='output{}'.format(idx + 1))
            xentropy = class_balanced_sigmoid_cross_entropy(
                b, heatmap,
                name='xentropy{}'.format(idx + 1))
            costs.append(xentropy)

        # some magic threshold
        pred = tf.cast(tf.greater(output, 0.5), tf.int32, name='prediction')
        wrong = tf.cast(tf.not_equal(pred, heatmap), tf.float32)
        wrong = tf.reduce_mean(wrong, name='train_error')

        if get_current_tower_context().is_training:
            wd_w = tf.train.exponential_decay(2e-4, get_global_step_var(),
                                              80000, 0.7, True)
            wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
            costs.append(wd_cost)

            add_param_summary(('.*/W', ['histogram']))   # monitor W
            self.cost = tf.add_n(costs, name='cost')
            add_moving_summary(costs + [wrong, self.cost])

    def get_gradient_processor(self):
        return [ScaleGradient([('convfcweight.*', 0.1), ('conv5_.*', 5)])]


def get_data(name):
    isTrain = name == 'train'
    ds = IdCard('val', DATA_DIR)

    class CropMultiple16(imgaug.ImageAugmentor):

        def _get_augment_params(self, img):
            newh = img.shape[0] // 16 * 16
            neww = img.shape[1] // 16 * 16
            assert newh > 0 and neww > 0
            diffh = img.shape[0] - newh
            h0 = 0 if diffh == 0 else self.rng.randint(diffh)
            diffw = img.shape[1] - neww
            w0 = 0 if diffw == 0 else self.rng.randint(diffw)
            return (h0, w0, newh, neww)

        def _augment(self, img, param):
            h0, w0, newh, neww = param
            return img[h0:h0 + newh, w0:w0 + neww]

    class Identity(imgaug.ImageAugmentor):

        def _get_augment_params(self, img):
            return None

        def _augment(self, img, param):
            return img

    if isTrain:
        shape_aug = [
            imgaug.RandomResize(xrange=(0.7, 1.5), yrange=(0.7, 1.5),
                                aspect_ratio_thres=0.15),
            imgaug.RotationAndCropValid(90),
            CropMultiple16(),
            imgaug.Flip(horiz=True),
            imgaug.Flip(vert=True)
        ]
    else:
        shape_aug = [Identity()]
    ds = AugmentImageComponents(ds, shape_aug, (0, 1))

    def f(m):
        m[m >= 0.50] = 1
        m[m < 0.50] = 0
        return m
    ds = MapDataComponent(ds, f, 1)

    if isTrain:
        augmentors = [
            imgaug.Brightness(63, clip=False),
            imgaug.Contrast((0.4, 1.5)),
        ]
        ds = AugmentImageComponent(ds, augmentors)
        ds = BatchDataByShape(ds, 8, idx=0)
        ds = PrefetchDataZMQ(ds, 1)
    else:
        ds = BatchData(ds, 1)
    return ds


def view_data():
    ds = RepeatedData(get_data('train'), -1)
    ds.reset_state()
    for ims, heatmaps in ds.get_data():
        for im, heatmap in zip(ims, heatmaps):
            assert im.shape[0] % 16 == 0 and im.shape[1] % 16 == 0, im.shape
            cv2.imshow("im", im / 255.0)
            cv2.waitKey(1000)
            cv2.imshow("edge", heatmap)
            cv2.waitKey(1000)


def get_config():
    logger.auto_set_dir()
    dataset_train = get_data('train')
    step_per_epoch = dataset_train.size() * 40
    dataset_val = get_data('val')

    lr = get_scalar_var('learning_rate', 3e-5, summary=True)
    return TrainConfig(
        dataflow=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
        callbacks=Callbacks([
            StatPrinter(), ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(30, 6e-6), (45, 1e-6), (60, 8e-7)]),
            HumanHyperParamSetter('learning_rate'),
            InferenceRunner(dataset_val,
                            BinaryClassificationStats('prediction', 'heatmap4d'))
        ]),
        model=Model(),
        step_per_epoch=step_per_epoch,
        max_epoch=100,
    )


def run(model_path, image_path, output):
    pred_config = PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_path),
        input_names=['image'],
        output_names=['output' + str(k) for k in range(1, 4)])
    predict_func = get_predict_func(pred_config)
    im = cv2.imread(image_path)
    assert im is not None
    im = cv2.resize(im, (im.shape[1] // 16 * 16, im.shape[0] // 16 * 16))
    outputs = predict_func([[im.astype('float32')]])
    if output is None:
        for k in range(3):
            pred = outputs[k][0]
            cv2.imwrite("out{}.png".format(
                '-fused' if k == 3 else str(k + 1)), pred * 255)
    else:
        pred = outputs[3][0]
        cv2.imwrite(output, pred * 255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--run', help='run model on images')
    parser.add_argument('--output', help='fused output filename. default to out-fused.png')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.view:
        view_data()
    elif args.run:
        run(args.load, args.run, args.output)
    else:
        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        if args.gpu:
            config.nr_tower = len(args.gpu.split(','))
        SyncMultiGPUTrainer(config).train()

