#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/4/4 21:32
# @Author: Aaron Meng
# @File  : relu.py
import numpy as np


class Relu(object):
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, eta):
        eta[self.x <= 0] = 0
        return eta


if __name__ == '__main__':
    pass
