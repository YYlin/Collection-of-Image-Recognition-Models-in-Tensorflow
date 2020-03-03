# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 20:33
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : dataload.py
import os
import cv2
import numpy as np
from keras.utils import to_categorical


def load_satetile_image(img_size=100, y_dim=10):
    img_list = []
    label_list = []
    dir_counter = 0

    path = 'data'

    # 对路径下的所有子文件夹中的所有jpg文件进行读取并存入到一个list中
    for child_dir in os.listdir(path):
        child_path = os.path.join(path, child_dir)
        for dir_image in os.listdir(child_path):
            img = cv2.imread(os.path.join(child_path, dir_image))
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
            img = img / 255.0
            img_list.append(img)
            label_list.append(dir_counter)

        dir_counter += 1

    X_train = np.array(img_list)
    Y_train = to_categorical(label_list, y_dim)
    return X_train, Y_train
