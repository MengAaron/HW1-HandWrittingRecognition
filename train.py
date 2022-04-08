#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/4/1 19:12
# @Author: Aaron Meng
# @File  : train.py
import os, argparse, yaml, easydict
from tqdm import tqdm, trange
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import util
import mnist
from util import optim


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    args = parser.parse_args()
    with open(args.cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    return args, easydict.EasyDict(cfg)


def plot(x):
    x = x.reshape([28, 28])
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.imshow(x, cmap='gray')


def save(parameters, path):
    data = dict()
    for name, para_list in parameters.items():
        for i, para in enumerate(para_list):
            data[name + '_' + str(i)] = para.data
    np.savez(path, **data)


def load(parameters, path):
    data = np.load(path)
    for name, para_list in parameters.items():
        for i in range(len(para_list)):
            parameters[name][i].data = data[name + '_' + str(i)]


def train(net, loss_fn, optimizer, batch_size, epoch=1, loadfile=None):
    X, Y = mnist.load(type='train')
    total_it_each_epoch = X.shape[0] // batch_size
    total_pos = 0
    if loadfile and os.path.isfile('para.npz'): load(net.parameters, 'para.npz')
    with trange(epoch) as tbar:
        for _ in tbar:
            total_pos = 0
            for i in tqdm(range(total_it_each_epoch)):
                start, end = i * batch_size, (i + 1) * batch_size
                x = X[start:end, :]
                y = Y[start:end]
                y_pred = net.forward(x)
                loss, acc = loss_fn.forward(y_pred, y)
                # print(loss)
                total_pos += acc * batch_size
                tbar.set_description(desc='正确率为%.5f' % (total_pos / ((i + 1) * batch_size)))
                tbar.refresh()
                eta = loss_fn.gradient()
                net.backward(eta)
                optimizer.update()
        # if i == 5000:
        #     import pdb
        #     pdb.set_trace()
    save(net.parameters, 'para.npz')


def test(net, loss_fn, batch_size):
    X, Y = mnist.load(type='test')
    total_it_each_epoch = X.shape[0] // batch_size
    total_pos = 0
    assert os.path.isfile('para.npz')
    load(net.parameters, 'para.npz')
    for i in tqdm(range(total_it_each_epoch)):
        start, end = i * batch_size, (i + 1) * batch_size
        x = X[start:end, :]
        y = Y[start:end]
        y_pred = net.forward(x)
        total_pos += np.sum(y == y_pred.argmax(axis=-1))
        print(total_pos / ((i + 1) * batch_size))


if __name__ == '__main__':
    # layers = [
    #     {'name': 'linear_1', 'type': 'linear', 'in_features': 784, 'out_features': 400},
    #     {'name': 'Relu1', 'type': 'Relu'},
    #     {'name': 'linear_2', 'type': 'linear', 'in_features': 400, 'out_features': 10},
    #     # {'name': 'Relu2', 'type': 'Relu'},
    #     # {'name': 'linear_3', 'type': 'linear', 'in_features': 100, 'out_features': 10},
    # ]
    args, cfg = parse_config()
    batch_size = cfg.batch_size
    loss_fn = util.CELoss()
    net = util.Net(cfg.MODEL)
    optimizer = optim.SGD(cfg.lr, net.parameters, cfg.decay, cfg.lr_decay)
    train(net, loss_fn, optimizer, batch_size, cfg.epoch, cfg.loadfile)
    # test(net, loss_fn, batch_size)
