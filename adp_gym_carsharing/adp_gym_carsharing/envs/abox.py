# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 10:40:07 2018

@author: Ibrahim
"""
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import gym
from gym import logger

class ABox(gym.Space):
    """
    A box in R^n.
    I.e., each coordinate is bounded.

    Example usage:
    self.action_space = spaces.Box(low=-10, high=10, shape=(1,))
    """
    def __init__(self, low=None, high=None, shape=None, dtype=None):
        """
        Two kinds of valid input:
            Box(low=-1.0, high=1.0, shape=(3,4)) # low and high are scalars, and shape is provided
            Box(low=np.array([-1.0,-2.0]), high=np.array([2.0,4.0])) # low and high are arrays of the same shape
        """
        if shape is None:
            assert low.shape == high.shape
            shape = low.shape
        else:
            assert np.isscalar(low) and np.isscalar(high)
            low = low + np.zeros(shape)
            high = high + np.zeros(shape)
        if dtype is None:  # Autodetect type
            if (high == 255).all():
                dtype = np.uint8
            else:
                dtype = np.float32
            logger.warn("gym.spaces.Box autodetected dtype as %s. Please provide explicit dtype." % dtype)
        self.low = low.astype(dtype)
        self.high = high.astype(dtype)
        gym.Space.__init__(self, shape, dtype)

    def sample(self):
       a1= gym.spaces.np_random.uniform(low=self.low, high=self.high)#.astype(int)# + (0 if self.dtype.kind == 'f' else 1), size=self.low.shape).astype(self.dtype)
#       a2= gym.spaces.np_random.uniform(low=self.low[int(len(self.low)/2):len(self.low)],high=state).astype(int)
       return a1 #np.concatenate((a1,a2))
       
    def contains(self, x):
        return   (x >= self.low).all() and (x <= self.high).all() 
    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()
    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return "Box" + str(self.shape)
    def __eq__(self, other):
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)