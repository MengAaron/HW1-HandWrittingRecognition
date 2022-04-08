#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/4/4 13:55
# @Author: Aaron Meng
# @File  : parameter.py
class Parameter(object):
    def __init__(self, data):
        self.data = data
        self.grad = None

    @property
    def T(self):
        return self.data.T

if __name__ == '__main__':
    pass
