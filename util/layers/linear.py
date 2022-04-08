#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/4/4 13:52
# @Author: Aaron Meng
# @File  : linear.py
import numpy as np
from ..parameter import Parameter


class Linear(object):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(np.random.randn(self.in_features, self.out_features) * 0.01)
        self.bias = Parameter(np.random.randn(self.out_features) * 0.01) if bias else None

    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        self.x = x
        x = np.dot(x, self.weights.data)
        if self.bias:
            x += self.bias.data
        return x

    def backward(self, eta):  # 输入为损失关于a_i 的梯度
        self.weights.grad = np.dot(self.x.T, eta)
        if self.bias:
            self.bias.grad = np.sum(eta, axis=0)
        return np.dot(eta, self.weights.T)


# 把对x的梯度回传

if __name__ == '__main__':
    pass
