#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/4/4 20:02
# @Author: Aaron Meng
# @File  : softmax.py
import numpy as np


class Softmax(object):
    def forward(self, x):
        x_max = np.max(x, axis=1, keepdims=True)
        x_exp = np.exp(x - x_max)
        self.prob = x_exp / np.sum(x_exp, axis=1, keepdims=True)
        self.prob[self.prob < 1e-6] = 1e-6
        return self.prob

    def backward(self, eta):
        pass


if __name__ == '__main__':
    pass
