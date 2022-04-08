#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/4/4 16:11
# @Author: Aaron Meng
# @File  : net.py
from .layers import *
from collections import defaultdict


class Net(object):
    def __init__(self, config):
        self.layers = []
        self.parameters = defaultdict(list)
        self.build_net(config)

    def build_net(self, config):
        for layer_config in config:
            self.layers.append(self.build_layer(layer_config))

    def build_layer(self, layer_config):
        layer_name = layer_config['name']
        layer_type = layer_config['type']
        if layer_type == 'linear':
            layer = Linear(**layer_config)
            self.parameters[layer_name].append(layer.weights)
            self.parameters[layer_name].append(layer.bias)
        elif layer_type == 'Relu':
            layer = Relu()
        else:
            raise NotImplementedError
        return layer

    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, eta):
        for layer in self.layers[::-1]:
            eta = layer.backward(eta)


if __name__ == '__main__':
    pass
