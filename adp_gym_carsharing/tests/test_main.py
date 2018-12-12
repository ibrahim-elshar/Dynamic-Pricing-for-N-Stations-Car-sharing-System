#!/usr/bin/env python
# -*- coding: utf-8 -*-

# core modules
import unittest

# 3rd party modules
import gym
import numpy as np

# internal modules
import gym_carsharing


class Environments(unittest.TestCase):

    def test_env(self):
        env = gym.make('Carsharing-v0')
        env.seed(0)
        env.reset()
        env.step(np.array([3.5, 4, 3.25, 5, 5.3]))
