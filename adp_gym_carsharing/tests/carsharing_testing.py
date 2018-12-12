# -*- coding: utf-8 -*-
"""
Created on Tue Jun 05 15:40:44 2018

@author: IJE8
"""

import gym
import gym.spaces
import adp_gym_carsharing
import numpy as np
env = gym.make('AdpCarsharing-v0')
#env = gym.make('CartPole-v0')


#ob=env.reset()
#
#action = env.action_space.sample(ob)
#observation, reward, done, info = env.step(action)
#print(observation, reward, done, info)

env.reset()
returns=0
for i in range(13):
#    print(env.observation)
    action = env.action_space.sample()#env.observation)
    print(action)
    observation, reward, done, info = env.step(action)
    print(observation, reward, done, info)
    returns+=reward
    print(i)
    print(returns)
    if done: 
        env.reset()


