# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 23:02:04 2021

@author: Asus
"""

import theano
from collections import OrderedDict
from utility import *

class Optimizer(object):
    def __init__(self, params=None):
        if params is None:
            return NotImplementedError()
        self.params = params

    def updates(self, loss=None):
        if loss is None:
            return NotImplementedError()

        self.updates = OrderedDict()
        self.gparams = [T.grad(loss, param) for param in self.params]


class RMSprop(Optimizer):
    def __init__(self, learningrate=0.0001, alpha=0.99, eps=1e-8, params=None):
        super(RMSprop, self).__init__(params=params)

        self.learningrate = learningrate
        self.alpha = alpha
        self.eps = eps

        self.mss = [buildsharedzeros(t.get_value().shape,'ms') for t in self.params]

    def updates(self, loss=None):
        super(RMSprop, self).updates(loss=loss)

        for ms, param, gparam in zip(self.mss, self.params, self.gparams):
            _ms = ms*self.alpha
            _ms += (1 - self.alpha) * gparam * gparam
            self.updates[ms] = _ms
            self.updates[param] = param - self.learningrate * gparam / T.sqrt(_ms + self.eps)

        return self.updates