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
#from . import mbox 
#from . import abox
import mbox 
import abox
from gym.utils import seeding


a_1=np.array([9., 12.]) #AA A_all_other
b_1=np.array([-2., -2.]) #AA A_all_other
eps_1=np.array([-5,-5])
###############
a_2=np.array([8., 11.])
b_2=np.array([-2., -2.])
eps_2=np.array([-5,-5])
###############
a_3=np.array([10., 13.])
b_3=np.array([-3., -3.])
eps_3=np.array([-5,-5])
###############
a_4=np.array([11., 15.])
b_4=np.array([-2., -4.])
eps_4=np.array([-5,-5])
###############
a_5=np.array([9., 14.])
b_5=np.array([-1., -3.])
eps_5=np.array([-5,-5])
###############
a_6=np.array([8., 16.])
b_6=np.array([-2., -2.])
eps_6=np.array([-5,-5])
###############
a_7=np.array([13., 18.])
b_7=np.array([-4., -3.])
eps_7=np.array([-5,-5])
###############
a_8=np.array([10., 14.])
b_8=np.array([-2., -2.])
eps_8=np.array([-5,-5])
###############
a_9=np.array([13., 15.])
b_9=np.array([-3., -2.])
eps_9=np.array([-5,-5])
###############
a_10=np.array([14., 16.])
b_10=np.array([-4., -2.])
eps_10=np.array([-5,-5])
###############################################################################
a=np.concatenate((a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10))
b=np.concatenate((b_1,b_2,b_3,b_4,b_5,b_6,b_7,b_8,b_9,b_10))
aa=np.array([a_1[0],a_2[0],a_3[0],a_4[0],a_5[0],a_6[0],a_7[0],a_8[0],a_9[0],a_10[0]]) # a 11 22 33 44 55 66 77 88 99 1010
ab=np.array([a_1[1],a_2[1],a_3[1],a_4[1],a_5[1],a_6[1],a_7[1],a_8[1],a_9[1],a_10[1]]) # a_to_other 1* 2* 3* 4* 5* 6* 7* 8* 9* 10*
bb=np.array([b_1[0],b_2[0],b_3[0],b_4[0],b_5[0],b_6[0],b_7[0],b_8[0],b_9[0],b_10[0]]) # b 11 22 33 44 55 66 77 88 99 1010
ba=np.array([b_1[1],b_2[1],b_3[1],b_4[1],b_5[1],b_6[1],b_7[1],b_8[1],b_9[1],b_10[1]]) # b_to_other 1* 2* 3* 4* 5* 6* 7* 8* 9* 10*

class AdpCarsharingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    N_def=10
    num_stages_def = 12
    MAX_CARS_def=100
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
    lost_sales_cost_def=np.empty(N_def*2); 
    lost_sales_cost_def.fill(6.0)#np.array([2, 2,2,2]) # ls_cost_AA ls_cost_BB ls_cost_AB ls_cost_BA

    # travel time estimate between stations
#    T_def=np.random.randint(1,4, size=(N_def, N_def))
#    for i in range(1, N_def):
#    for j in range(1,i+1):
#        T_def[i][j-1]=T_def[j-1][i]
    T_def=np.array([[1, 2, 1, 2, 2, 2, 1, 1, 3, 2],
                    [2, 1, 2, 3, 2, 3, 3, 1, 2, 2],
                    [1, 2, 1, 3, 3, 2, 2, 1, 2, 1],
                    [2, 3, 3, 1, 2, 1, 3, 2, 3, 2],
                    [2, 2, 3, 2, 1, 3, 2, 1, 2, 3],
                    [2, 3, 2, 1, 3, 1, 1, 2, 1, 3],
                    [1, 3, 2, 3, 2, 1, 1, 3, 2, 2],
                    [1, 1, 1, 2, 1, 2, 3, 1, 3, 1],
                    [3, 2, 2, 3, 2, 1, 2, 3, 1, 3],
                    [2, 2, 1, 2, 3, 3, 2, 1, 3, 1]])

#    alpha=np.ones(shape=(N_def,N_def))
#    alpha=np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
#    prob_ij_def=np.random.dirichlet(np.ones(N_def),size=N_def)
    prob_ij_def=np.array([[0.18000618, 0.26759478, 0.04259375, 0.17820878, 0.02228519,
                        0.01863454, 0.08354288, 0.14278845, 0.01383538, 0.05051007],
                       [0.14495372, 0.05357948, 0.02155977, 0.03694435, 0.00415032,
                        0.09027639, 0.26778211, 0.23402727, 0.0600922 , 0.08663439],
                       [0.12064387, 0.05629082, 0.20467411, 0.16815461, 0.03158785,
                        0.09984876, 0.09170807, 0.00913912, 0.0789744 , 0.1389784 ],
                       [0.03954417, 0.02466245, 0.04729991, 0.15920888, 0.05473802,
                        0.06869946, 0.35490087, 0.03827747, 0.09455603, 0.11811275],
                       [0.03504996, 0.29565191 , 0.28662327, 0.02103628, 0.06984387,
                        0.05203944, 0.0869573 , 0.02749044, 0.05763857, 0.06766896],
                       [0.0708694 , 0.02130604, 0.06527834, 0.10476633, 0.38887485,
                        0.00566306, 0.06975661, 0.04304322, 0.16952313, 0.06091902],
                       [0.08119415, 0.18705128, 0.00639008, 0.06072982, 0.0734726 ,
                        0.00792236, 0.27340249, 0.20536148, 0.01619699, 0.08827874],
                       [0.22703236, 0.0474092 , 0.01752548, 0.10744341, 0.0838506 ,
                        0.00444494, 0.08865416, 0.1930754 , 0.01813294, 0.21243149],
                       [0.18208531, 0.244607  , 0.14186909, 0.11135224, 0.07362299,
                        0.01924541, 0.00930965, 0.06104104, 0.0012221 , 0.15564517],
                       [0.32127767, 0.1543693 , 0.01833383, 0.08735162, 0.0608624 ,
                        0.02074942, 0.02128457, 0.05921182, 0.11077644, 0.14578292]])
#    prob_ij_def=np.ones(shape=(N_def,N_def))
#    for i in range(N_def):
#        prob_ij_def[i]=np.random.dirichlet(alpha[i],size=1)[0]
##    prob_ij_def=np.random.dirichlet(alpha,size=N_def)
##    prob_ij_def=np.zeros(shape=(N_def,N_def))
##    print(prob_ij_def)
    i=0
    for row in prob_ij_def:
##        print(row)
        row[i]=0
        if i==len(row): 
            row[-1]=1+row[-1]-sum(row) 
        else:
            row[-2]=1+row[-2]-sum(row)
#        print(row)
##        row=np.delete(row,-1)
##        print(row)
        prob_ij_def[i]=row
        i +=1
#    print(prob_ij_def)
#    prob_ij_def[i]=np.random.dirichlet(np.ones(N_def),size=10)

    def __init__(self, num_stations=N_def, num_cars=MAX_CARS_def, aa_min_price=aa_pmin_def, aa_max_price=aa_pmax_def,ab_min_price=ab_pmin_def, ab_max_price=ab_pmax_def, aa=aa_def, ab=ab_def,bb=bb_def, ba=ba_def, discount_rate=discount_rate_def, num_stages=num_stages_def, aa_epsSupL=aa_epsSupL_def,aa_epsSupH=aa_epsSupH_def,ab_epsSupL=ab_epsSupL_def,ab_epsSupH=ab_epsSupH_def,travel_time_btwn_stat=T_def,prob_ij=prob_ij_def,lost_sales_cost=lost_sales_cost_def):
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
        self.prob_ij=prob_ij
        self.lost_sales_cost=lost_sales_cost
        self.discount_rate=discount_rate
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
        # theta !!
        w=np.minimum(demand + epsVector, thetas).astype(float)
#        print("w="+str(w))
        wij=np.zeros((self.N,self.N))
        for j in range(self.N, (self.N*2)):
#                print("j="+str(j))             
                if w[j]!=0:
                    wij[j-self.N,:]=np.random.multinomial(w[j], self.prob_ij[j-self.N], size=1)[0] #mbox.arandomize(w[j], self.N-1,j-self.N) 
                    #wij=[w_j1 w_j2 w_j3 w_j4 ... w_ji... w_jN]
                else:
                    wij[j-self.N,:]=np.zeros(self.N)
#        print("wij="+str(wij))
#        print("size wij="+str(wij.shape))
#        reward =np.around(np.dot(w , price), 2)
#        print("T="+str(self.T))
        num_lost_sales = demand +epsVector - w
        
        Twij=np.multiply(self.T, wij)    
#        print("Twij="+str(Twij))
#        print("np.sum(Twij, axis=1)="+str(np.sum(Twij, axis=1)))
#        reward =np.around(sum(np.sum(Twij, axis=1) * price[self.N:self.N*2]) + (np.dot(w[0:self.N] , price[0:self.N])), 2)
        ls_cost_ow=np.dot(num_lost_sales[self.N:self.N*2] ,  self.lost_sales_cost[self.N:self.N*2])
        ls_cost_rt=np.dot(num_lost_sales[0:self.N] ,  self.lost_sales_cost[0:self.N])
#        print("ls_cost_ow="+str(ls_cost_ow))
#        print("ls_cost_rt="+str(ls_cost_rt))
        profit_ow=np.dot(np.sum(Twij, axis=1) , price[self.N:self.N*2] )
        profit_rt=np.dot(w[0:self.N] , price[0:self.N])
#        print("profit_ow="+str(profit_ow))
#        print("profit_rt="+str(profit_rt))
        reward =np.around(profit_ow + profit_rt - ls_cost_ow - ls_cost_rt,2)
#        reward =np.around(sum(np.sum(Twij, axis=1) * price[self.N:self.N*2] - (num_lost_sales[self.N:self.N*2] *  self.lost_sales_cost[self.N:self.N*2])) + (np.dot(w[0:self.N] , price[0:self.N])- (num_lost_sales[0:self.N] *  self.lost_sales_cost[0:self.N])), 2)
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

        
        