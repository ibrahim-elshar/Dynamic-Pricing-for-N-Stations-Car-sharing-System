# -*- coding: utf-8 -*-
"""
Created on Wed Jun 06 18:36:28 2018

@author: Ibrahim
"""
import gym
import numpy as np
from gym import spaces
#import random
import logging.config
import mbox 
from gym.utils import seeding


class CarsharingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    N_def=5
    MAX_CARS_def=50
    pmin_def=np.empty(N_def); pmin_def.fill(3.0)
#    T_def=np.random.randint(1,4, size=(N_def, N_def))
#    T_def=np.ones((N_def, N_def)).astype(int)
    T_def=np.array([[2, 2, 3, 1, 3],
           [2, 3, 3, 1, 2],
           [3, 1, 2, 2, 2],
           [2, 3, 2, 2, 3],
           [1, 2, 3, 2, 2]])
#    a_def=np.random.randint(30, 45, N_def).astype(float)
#    b_def=np.random.randint(-5, -1, N_def).astype(float)
#    a_def=np.array([30., 30.])
#    b_def=np.array([-5., -5.])
    a_def=np.array([31., 38., 37., 31., 42.])
    b_def=np.array([-3., -4., -5., -5., -3.])
    def __init__(self, num_stations=N_def, num_cars=MAX_CARS_def, min_price=pmin_def, travel_time_btwn_stat=T_def, a=a_def, b=b_def):
        self.__version__ = "0.1.0"
        logging.info("CarsharingEnv - Version {}".format(self.__version__))
        # General variables defining the environment
        self.N = num_stations  # Number of stations
        self.MAX_CARS = num_cars # Max number of cars in the system 
        self.set_pmin= min_price
#        self.T = np.array([ ])
#        self.T=np.random.randint(1,4, size=(self.N, self.N))
        self.T=travel_time_btwn_stat
        self.kmax=np.max(self.T)
        self.a=a
        self.b=b
        self.pmin=min_price
        self.pmax=np.empty(self.N); self.pmax= self.a / (-1* self.b) -0.1 # -1 set the price such that the expected demand cannot be negative
        self.pmax=np.around(self.pmax, 1)
        self.dmin=self.D(self.pmax)
        self.dmax=self.D(self.pmin)
        self.Nik= []
        for i in range(self.N):
            self.Nik.append([np.where(self.T[:,i] == k) for k in range(1, self.kmax+1)])
        self.action_space = spaces.Box(self.pmin, self.pmax, dtype=np.float32)
        self.s=spaces.Box(0, self.kmax,shape=(self.N,self.kmax), dtype=np.uint8)
        self.x=mbox.MBox(self.MAX_CARS, self.N)
        self.observation_space = spaces.Tuple((self.x,self.s))        
        self.t=0
        
    #expected demand function 
    def D(self, p):
      for i in range(self.N):
       if p[i] >= self.pmax[i]:
          p[i]=self.pmax[i]
       elif p[i] <= self.pmin[i]:
          p[i]=self.pmin [i] 
      d=np.empty(self.N)
      d= self.a + self.b*p
      #######
      #d=np.rint(d).astype(int)
      #######
      return d
    # inverse of expected demand function; returns the price for a given expected demand vector.
    def P(self, d):
      for i in range(self.N):
       if d[i] >= self.dmax[i]:
          d[i]=self.dmax[i]
       elif d[i] <= self.dmin[i]:
            d[i]=self.dmin[i] 
      p=(d-self.a)/self.b
      p=np.around(p, 1)
      return p
      
    
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
        demand=np.rint(self.D(action)).astype(int)
        #demand=self.D(action)
        low=-self.D(action)
        low=low.astype(int)
        high=-low+1
        epsVector=[]
        for i in range(self.N):
           epsVector.append(np.random.randint(low[i],high[i])) 
        newx=[]
        w=np.minimum(demand + epsVector, self.x)
        wij=np.zeros((self.N,self.N))
        for j in range(self.N):
                if w[j]!=0:
                    wij[j,:]=mbox.randomize(w[j], self.N) #wij=[w_j1 w_j2 w_j3 w_j4 ... w_ji... w_jN]
                else:
                    wij[j,:]=np.zeros(self.N)
        Twij=np.multiply(self.T, wij)    
        reward =np.around(sum(np.sum(Twij, axis=1) * action), 2)
#        newx = self.x  +self.s[:,0] - w
        temp=np.zeros(self.N)
        for i in range(self.N):
            temp[i]=np.sum([wij[j,i] for j in self.Nik[i][0]])
        newx = self.x  +self.s[:,0] +temp - w
        news=np.zeros((self.N,self.kmax))
        if self.kmax>1:
            for i in range(self.N):
                for k in range(self.kmax):
    #                print("k="+str(k))
                    if k==0:
                        news[i,k]= self.s[i,k+1] 
                    elif k==(self.kmax-1):
                        news[i,k]=np.sum([wij[j,i] for j in self.Nik[i][k]])                                 
                    else:  
#                        print("k="+str(k))
                        news[i,k]=np.sum([wij[j,i] for j in self.Nik[i][k]]) + self.s[i,k+1]
#        print("newS="+str(newS))
        self.x=newx
        self.s=news
        ob= (self.x, self.s)
        self.t +=1
        if self.t>=12:
            done = True
        else:
            done =  False
        
        return ob, reward, done, {str(self.t)}

    def reset(self):
        self.s=np.zeros((self.N, self.kmax))
        self.x=mbox.randomize(self.MAX_CARS, self.N)
        ob = (self.x, self.s)
        self.t=0
        return ob

    def render(self, mode='human', close=False):
        pass


    def _seed(self, seed=None):
         self.np_random, seed = seeding.np_random(seed)
         return [seed]