# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 20:02
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : main.py
import argparse
import sys
import tensorflow as tf
from Model import Model


# parsing and configuration
def parse_args():
    desc = "Tensorflow implementation of StarGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='satellite', help='dataset_name')
    parser.add_argument('--type_of_model', type=str, default='inception_V4', help='ResNet Alexnet VGG19 inception_V4')

    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--img_size', type=int, default=100, help='The size of image')
    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch size')
    parser.add_argument('--y_dim', type=int, default=9, help='The classification of dataset')

    parser.add_argument('--resnet_type', type=str, default='50', help='The 50 101 152 layer can be chose')

    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        print('The args is None')
        sys.exit()
    else:
        print('args:', args)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if args.type_of_model == 'Alexnet':
            print('******  Using Alexnet ******')
            model = Model(sess, type_of_model=args.type_of_model, epoch=args.epoch, dataset_name=args.dataset, batch_size=args.batch_size,
                          img_size=args.img_size, y_dim=args.y_dim, resnet_type=args.resnet_type)
        elif args.type_of_model == 'ResNet':
            print('******  Using ResNet ******')
            model = Model(sess, type_of_model=args.type_of_model, epoch=args.epoch, dataset_name=args.dataset,
                          batch_size=args.batch_size,
                          img_size=args.img_size, y_dim=args.y_dim, resnet_type=args.resnet_type)
        elif args.type_of_model == 'VGG19':
            print('******  Using VGG19 ******')
            model = Model(sess, type_of_model=args.type_of_model, epoch=args.epoch, dataset_name=args.dataset,
                          batch_size=args.batch_size,
                          img_size=args.img_size, y_dim=args.y_dim, resnet_type=args.resnet_type)
        elif args.type_of_model == 'inception_V4':
            print('******  Using inception_V4 ******')
            model = Model(sess, type_of_model=args.type_of_model, epoch=args.epoch, dataset_name=args.dataset,
                          batch_size=args.batch_size,
                          img_size=args.img_size, y_dim=args.y_dim, resnet_type=args.resnet_type)
        else:
            print('these is no %s'%args.type_of_model)
            sys.exit()

        # build graph
        model.build_model()

        model.train()


if __name__ == '__main__':
    main()

