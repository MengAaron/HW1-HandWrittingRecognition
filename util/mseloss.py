#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/4/4 10:19
# @Author: Aaron Meng
# @File  : mseloss.py
import numpy as np


class MSELoss(object):
    def forward(self, pred, y):
        y_onehot = np.zeros_like(pred)
        y_onehot[y] = 1.0
        self.diff = pred - y_onehot
        return np.dot(self.diff.T, self.diff)
    def gradient(self):
        return self.diff

if __name__ == '__main__':
    pass
