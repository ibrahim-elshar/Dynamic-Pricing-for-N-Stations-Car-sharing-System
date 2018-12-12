# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 12:50:24 2018

@author: IJE8
"""

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
import abox
from gym.utils import seeding


###############################################################################
a_1=np.array([10., 15.]) #AA A_all_other
b_1=np.array([-2., -2.]) #AA A_all_other
###############
a_2=np.array([11., 11.])
b_2=np.array([-2., -2.])
###############
a_3=np.array([12., 12.])
b_3=np.array([-3., -3.])
###############
a_4=np.array([11., 14.])
b_4=np.array([-2., -4.])
###############
a_5=np.array([13., 17.])
b_5=np.array([-1., -3.])
###############
a_6=np.array([14., 16.])
b_6=np.array([-2., -2.])
###############
a_7=np.array([13., 17.])
b_7=np.array([-4., -3.])
###############
a_8=np.array([10., 18.])
b_8=np.array([-2., -2.])
###############
a_9=np.array([10., 16.])
b_9=np.array([-3., -2.])
###############
a_10=np.array([12., 16.])
b_10=np.array([-4., -2.])
###############################################################################
a=np.concatenate((a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10))
b=np.concatenate((b_1,b_2,b_3,b_4,b_5,b_6,b_7,b_8,b_9,b_10))
aa=np.array([a_1[0],a_2[0],a_3[0],a_4[0],a_5[0],a_6[0],a_7[0],a_8[0],a_9[0],a_10[0]]) # a 11 22 33 44 55 66 77 88 99 1010
ab=np.array([a_1[1],a_2[1],a_3[1],a_4[1],a_5[1],a_6[1],a_7[1],a_8[1],a_9[1],a_10[1]]) # a_to_other 1* 2* 3* 4* 5* 6* 7* 8* 9* 10*
bb=np.array([b_1[0],b_2[0],b_3[0],b_4[0],b_5[0],b_6[0],b_7[0],b_8[0],b_9[0],b_10[0]]) # b 11 22 33 44 55 66 77 88 99 1010
ba=np.array([b_1[1],b_2[1],b_3[1],b_4[1],b_5[1],b_6[1],b_7[1],b_8[1],b_9[1],b_10[1]]) # b_to_other 1* 2* 3* 4* 5* 6* 7* 8* 9* 10*

class CarsharingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    N_def=10
    num_stages_def = 6
    MAX_CARS_def=50
#    pmin_def=np.empty(N_def*2); pmin_def.fill(1.0)
#    pmax_def=np.empty(N_def*2); pmax_def.fill(10.0)
#    a_def=np.concatenate((a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10))
#    b_def=np.concatenate((b_1,b_2,b_3,b_4,b_5,b_6,b_7,b_8,b_9,b_10))
    #return demand model
    aa_def=aa
    bb_def=bb
    
    #one way demand model
    ab_def=ab
    ba_def=ba
    
    #return demand models
    aa_pmin_def=np.empty(N_def); aa_pmin_def.fill(1.0)
    aa_pmax_def=np.empty(N_def); aa_pmax_def.fill(10.0)
    
    #one way demand models
    ab_pmin_def=np.empty(N_def); ab_pmin_def.fill(1.0)
    ab_pmax_def=np.empty(N_def); ab_pmax_def.fill(10.0)
#    epsSupL_def=np.array([-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5])
#    epsSupH_def=-epsSupL_def
    
    #return demand models epsilon support
    aa_epsSupL_def=np.array([-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0])
    aa_epsSupH_def=-aa_epsSupL_def
    
    #one way demand models epsilon support
    ab_epsSupL_def=np.array([-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0])
    ab_epsSupH_def=-ab_epsSupL_def
    
    discount_rate_def=0.99
    
    # travel time estimate between stations
#    T_def=np.random.randint(1,4, size=(10, 10))
    T_def=np.array([[3, 2, 1, 2, 2, 2, 1, 1, 3, 2],
                    [3, 3, 2, 3, 2, 3, 3, 1, 2, 2],
                    [1, 2, 2, 3, 3, 2, 2, 1, 2, 1],
                    [3, 3, 1, 2, 2, 1, 3, 2, 3, 2],
                    [2, 3, 3, 1, 1, 3, 2, 1, 2, 3],
                    [1, 1, 1, 1, 3, 1, 1, 2, 1, 3],
                    [1, 3, 3, 2, 2, 2, 2, 3, 2, 2],
                    [2, 3, 2, 1, 1, 3, 3, 3, 3, 1],
                    [1, 2, 3, 1, 2, 2, 3, 1, 2, 3],
                    [3, 3, 3, 2, 1, 3, 2, 1, 1, 1]])

    def __init__(self, num_stations=N_def, num_cars=MAX_CARS_def, aa_min_price=aa_pmin_def, aa_max_price=aa_pmax_def,ab_min_price=ab_pmin_def, ab_max_price=ab_pmax_def, aa=aa_def, ab=ab_def,bb=bb_def, ba=ba_def, discount_rate=discount_rate_def, num_stages=num_stages_def, aa_epsSupL=aa_epsSupL_def,aa_epsSupH=aa_epsSupH_def,ab_epsSupL=ab_epsSupL_def,ab_epsSupH=ab_epsSupH_def,travel_time_btwn_stat=T_def):
        self.__version__ = "0.1.0"
        logging.info("CarsharingEnv - Version {}".format(self.__version__))   
        self.N = num_stations  # Number of stations
        self.MAX_CARS = num_cars # Max number of cars in the system 
        
        #return trip demand model
        self.aa=aa
        self.bb=bb
        
        #one way trip demand model
        self.ab=ab
        self.ba=ba
        
        #return trips epsilon support
        self.aa_epsSupL=aa_epsSupL
        self.aa_epsSupH=aa_epsSupH
        
        #one trips epsilon support
        self.ab_epsSupL=ab_epsSupL
        self.ab_epsSupH=ab_epsSupH
        
        #return min/max price
        self.aa_pmin= aa_min_price
        self.aa_pmax=np.minimum(np.around((self.aa +self.aa_epsSupL) / (-1* self.bb) , 1), aa_max_price)
        
        #return fixed demand and price for one-way station
        self.aa_p= (self.aa_pmin+self.aa_pmax)*0.5
        self.aa_d= self.DAA(self.aa_p)
        
        
        #one_way trips min/max price
        self.ab_pmin= ab_min_price
        self.ab_pmax=np.minimum(np.around((self.ab +self.ab_epsSupL) / (-1* self.ba) , 1), ab_max_price)
           
        
        self.aa_dmin=self.DAA(self.aa_pmax)
        self.aa_dmax=self.DAA(self.aa_pmin)
        
        self.ab_dmin=self.DAB(self.ab_pmax)
        self.ab_dmax=self.DAB(self.ab_pmin)
        
        #for the actions it is better to separate AA and A_all_other, since AAs are fixed
        # theta variables !!
        #easy need to create a variable 
        self.action_L=np.concatenate((self.ab_dmin, np.zeros(self.N)))
        self.action_H=np.concatenate((self.ab_dmax, np.multiply(np.ones(self.N), self.MAX_CARS)))
        self.action_space = abox.ABox(self.action_L, self.action_H, dtype=np.float32)
#        self.s=spaces.Box(0, self.kmax,shape=(self.N,self.kmax), dtype=np.uint8)
#        self.observation_space = spaces.Tuple((self.x,self.s))
        self.observation= np.multiply(np.ones(self.N), self.MAX_CARS/self.N)#mbox.randomize(self.MAX_CARS, self.N)
        self.observation_space = mbox.MBox(self.MAX_CARS, self.N)
        self.t=0
        self.T=travel_time_btwn_stat
        
    #expected demand function 
    #return trip
    def DAA(self, p):
      d= np.rint(self.aa + self.bb*p).astype(int)
      return d 
    #one way trip
    def DAB(self, p):
      d= np.rint(self.ab + self.ba*p).astype(int)
      return d 
  
    # inverse of expected demand function; returns the price for a given expected demand vector
    #return trip
    def PAA(self, d):
      p=np.around((d-self.aa)/self.bb, 1)
      return p
    #one way trip
    def PAB(self, d):
      p=np.around((d-self.ab)/self.ba, 1)
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
#        print("state="+str(self.observation))
#        print("action="+str(action))
        assert self.action_space.contains(action,self.observation) 
        price=np.concatenate((self.aa_p,self.PAB(action[0:self.N])))
#        print("price="+str(price))
        demand=np.concatenate((self.aa_d,action[0:self.N]))
#        print("demand="+str(demand))
        ab_theta=action[self.N:self.N*2]
        aa_theta=self.observation - ab_theta
        thetas=np.concatenate((aa_theta,ab_theta))
#        print("thetas="+str(thetas))
        epsVector=[]
        for i in range(self.N):
           epsVector.append(np.random.randint(self.aa_epsSupL[i],self.aa_epsSupH[i]+1))
        for i in range(self.N):
           epsVector.append(np.random.randint(self.ab_epsSupL[i],self.ab_epsSupH[i]+1))
#        print("epsVector="+str(epsVector))
        global w
        # theta !!
        w=np.minimum(demand + epsVector, thetas).astype(float)
#        print("w="+str(w))
        wij=np.zeros((self.N,self.N))
        for j in range(self.N, (self.N*2)):
#                print("j="+str(j))             
                if w[j]!=0:
                    wij[j-self.N,:]=mbox.arandomize(w[j], self.N-1,j-self.N) #wij=[w_j1 w_j2 w_j3 w_j4 ... w_ji... w_jN]
                else:
                    wij[j-self.N,:]=np.zeros(self.N)
#        print("wij="+str(wij))
#        print("size wij="+str(wij.shape))
#        reward =np.around(np.dot(w , price), 2)
        Twij=np.multiply(self.T, wij)    
        reward =np.around(sum(np.sum(Twij, axis=1) * price), 2)
#        print("reward="+str(reward))
        temp=np.sum(wij, axis=0)
#        print("temp="+str(temp))
        newState = self.observation  + temp - w[self.N: self.N*2]
#        print("newState="+str(newState))
        ob= newState
        self.observation=ob
        self.t +=1
        if self.t>=12:
            done = True
        else:
            done =  False
        
        return ob, reward, done, {str(self.t)}

       
    def reset(self):
        self.observation=np.multiply(np.ones(self.N), self.MAX_CARS/self.N) #mbox.randomize(self.MAX_CARS, self.N)
        self.t=0
        return self.observation

        
        