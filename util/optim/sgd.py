#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/4/4 21:55
# @Author: Aaron Meng
# @File  : sgd.py
class SGD(object):
    def __init__(self, lr, parameters, decay=0, lr_decay=None, lr_decay_step=5000):
        self.lr = lr
        self.para = parameters
        self.decay = decay
        self.total_iter = 0
        self.lr_decay = lr_decay
        self.lr_decay_step = lr_decay_step

    def update(self):
        self.total_iter += 1
        if self.lr_decay and (self.total_iter+1) % self.lr_decay_step == 0:
            self.lr *= self.lr_decay
        for para_list in self.para.values():
            for para in para_list:
                para.data -= self.lr * para.grad
                para.data *= (1-self.decay)


if __name__ == '__main__':
    pass
