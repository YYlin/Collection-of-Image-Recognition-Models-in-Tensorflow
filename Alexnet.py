# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 19:59
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : Alexnet.py
import tensorflow as tf


# 定义一个最大池化层
def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding="SAME"):
    return tf.nn.max_pool(x, ksize=[1, kHeight, kWidth, 1],
                          strides=[1, strideX, strideY, 1], padding=padding, name=name)


# 使用dropout 随机丢包
def dropout(x, keepPro, name=None):
    return tf.nn.dropout(x, keepPro, name)


# 定义一个局部正则化
def LRN(x, r, alpha, beta, name=None, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=r, alpha=alpha, beta=beta, bias=bias, name=name)


# 定义一个全连接层的函数 其实可以修改成之前我写的那个
def fcLayer(x, inputD, outputD, reluFlag, name):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[inputD, outputD], dtype="float")
        b = tf.get_variable("b", [outputD], dtype="float")
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)

        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out


def convLayer(x, kHeight, kWidth, strideX, strideY, featureNum, name, padding="SAME", groups=1):
    channel = int(x.get_shape()[-1])
    conv = lambda a, b: tf.nn.conv2d(a, b, strides=[1, strideY, strideX, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[kHeight, kWidth, channel / groups, featureNum])
        b = tf.get_variable("b", shape=[featureNum])

        xNew = tf.split(value=x, num_or_size_splits=groups, axis=3)
        wNew = tf.split(value=w, num_or_size_splits=groups, axis=3)

        featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
        mergeFeatureMap = tf.concat(axis=3, values=featureMap)
        # print mergeFeatureMap.shape
        out = tf.nn.bias_add(mergeFeatureMap, b)
        return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name=scope.name)


# 定义一个alexNet类别 方便之后调用
class AlexNet(object):
    def __init__(self, x, classNum, keepPro):
        self.X = x
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        self.begin_alexNet()

    def begin_alexNet(self):
        conv1 = convLayer(self.X, 11, 11, 4, 4, 96, "conv1", "VALID")
        lrn1 = LRN(conv1, 2, 2e-05, 0.75, "norm1")
        pool1 = maxPoolLayer(lrn1, 3, 3, 2, 2, "pool1", "VALID")

        conv2 = convLayer(pool1, 5, 5, 1, 1, 256, "conv2", groups=2)
        lrn2 = LRN(conv2, 2, 2e-05, 0.75, "lrn2")
        pool2 = maxPoolLayer(lrn2, 3, 3, 2, 2, "pool2", "VALID")

        conv3 = convLayer(pool2, 3, 3, 1, 1, 384, "conv3")

        conv4 = convLayer(conv3, 3, 3, 1, 1, 384, "conv4", groups=2)

        conv5 = convLayer(conv4, 3, 3, 1, 1, 256, "conv5", groups=2)
        pool5 = maxPoolLayer(conv5, 3, 3, 2, 2, "pool5", "VALID")

        fcIn = tf.reshape(pool5, [-1, 256 * 2 * 2])
        fc1 = fcLayer(fcIn, 256 * 2 * 2, 4096, True, "fc6")
        dropout1 = dropout(fc1, self.KEEPPRO)

        fc2 = fcLayer(dropout1, 4096, 4096, True, "fc7")
        dropout2 = dropout(fc2, self.KEEPPRO)

        self.fc3 = fcLayer(dropout2, 4096, self.CLASSNUM, True, "fc8")

