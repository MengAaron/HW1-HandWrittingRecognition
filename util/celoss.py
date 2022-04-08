#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/4/4 21:13
# @Author: Aaron Meng
# @File  : celoss.py

import numpy as np
from .layers import Softmax


class CELoss(object):
    def __init__(self):
        self.classifer = Softmax()

    def forward(self, pred, y):
        y_onehot = np.zeros_like(pred)
        y_onehot[np.arange(y.size), y] = 1.0
        pred_prob = self.classifer.forward(pred)
        self.diff = pred_prob - y_onehot
        acc = (pred_prob.argmax(axis=-1) == y_onehot.argmax(axis=-1)).mean()
        # import pdb
        # pdb.set_trace()
        return -np.sum(np.log(pred_prob)*y_onehot,axis=-1).mean(), acc

    def gradient(self):
        return self.diff


if __name__ == '__main__':
    pass
