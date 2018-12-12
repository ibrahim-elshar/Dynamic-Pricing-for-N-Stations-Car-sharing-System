# -*- coding: utf-8 -*-
'''
This file implements a vehicle sharing simulator.
The environment consists of stations (by default 5 stations) from which cars are rented in accordance to a price-demand model with some noise.
After setting the price at each station the demand is observed and the destination stations are randomly assigned.
The time until arrival is proportional to the distance between the origin and destination stations.
The objective is to set the rental prices at each station during each period to maximize the total revenue.
An episode is 12 periods long.
'''
import gym
import numpy as np
from gym import spaces
import sys
if sys.version_info[0] < 3:
    #raise Exception("Must be using Python 3")
    import mbox 
    import abox
    from Stations_Config import Stations
else:
    from . import mbox
    from . import abox
    from .Stations_Config import Stations
from gym.utils import seeding



class AdpCarsharingEnv(gym.Env):
    '''
    Creates the AdpCarsharingEnv.
    '''
    def __init__(self, num_stages = 12,\
                 ):
        self.num_stages = num_stages
        self.stations =  Stations()
        self.action_L=self.stations.pmin
        self.action_H=self.stations.pmax
        self.action_space = abox.ABox(self.action_L, self.action_H, dtype=np.float64)
        self.observation_space = mbox.MBox(self.stations.num_cars, self.stations.num_stations)
        #self.observation= np.multiply(np.ones(self.stations.num_stations), self.stations.num_cars/self.num_stations)
        self.t=0
        
    def step(self, action):
        """
        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        assert self.action_space.contains(action)
        price = action
        demand = self.stations.D(price)
        epsVector=[]
        for i in range(self.stations.num_stations):
           epsVector.append(np.random.randint(-self.stations.epsilons_support[i],self.stations.epsilons_support[i]+1))
#        print("epsVector=", epsVector)   
        w = np.minimum(demand + epsVector, self.observation).astype(float)
#        print("w=",w)
        wij=np.zeros((self.stations.num_stations,self.stations.num_stations))
        for j in range(self.stations.num_stations):        
                if w[j]!=0:
                    wij[j,:]=np.random.multinomial(w[j], self.stations.prob_ij[j], size=1)[0] 
                else:
                    wij[j,:]=np.zeros(self.stations.num_stations)
#        print("wij=",wij)           
        num_lost_sales = demand +epsVector - w
#        print ("num_lost_sales=", num_lost_sales)
        dwij=np.multiply(self.stations.distance_ij, wij)
#        print ("dwij=", dwij)
        lost_sales_cost=np.dot(num_lost_sales ,  self.stations.lost_sales_cost)
#        print ("lost_sales_cost=", lost_sales_cost)
        profit = np.dot(np.sum(dwij, axis=1) , price)
#        print("profit=", profit)
        reward = np.around(profit - lost_sales_cost, 2)
        new_observation = self.observation  + np.sum(wij, axis=0) - w
#        print("new_observation=",new_observation)  
#        print("t", self.t)
        self.observation = new_observation
        self.t +=1
        if self.t>=self.num_stages:
            done = True
        else:
            done =  False
        return self.observation, reward, done, {str(self.t)}
    
    def reset(self):
        self.observation=np.multiply(np.ones(self.stations.num_stations), self.stations.num_cars/self.stations.num_stations) #mbox.randomize(self.MAX_CARS, self.N)
        self.t=0
        return self.observation
                 
    def render(self, mode='human', close=False):
        pass


    def seed(self, seed=None):
         self.np_random, seed = seeding.np_random(seed)
         return [seed]

