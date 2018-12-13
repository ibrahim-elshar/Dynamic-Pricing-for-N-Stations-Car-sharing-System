# -*- coding: utf-8 -*-
"""
This file implements DP to solve each combination of 2 station pricing problem of the 5 stations.
The environment consists of stations (by default 5 stations) from which cars are rented in accordance to a price-demand model with some noise.
After setting the price at each station the demand is observed and the destination stations are randomly assigned.
The time until arrival is proportional to the distance between the origin and destination stations.
The objective is to set the rental prices at each station during each period to maximize the total revenue.
An episode is 12 periods long.
"""
import sys
if sys.version_info[0] < 3:
    from adp_gym_carsharing.envs.Stations_Config import Stations
else:
    from adp_gym_carsharing.envs.Stations_Config import Stations

import numpy as np
import itertools
import timeit
import matplotlib.pyplot as plt
import scipy.stats as ss
import os
import pickle

stations = Stations()
num_stages_def = 13
discount_rate_def = 1

def create_2_stations_problem(separate_station_num):
    '''
    TODO
    '''    
    idx = separate_station_num
    demand_par_a = np.array([ stations.demand_par_a[idx]  ,\
                             sum(stations.demand_par_a) -\
                             stations.demand_par_a[idx] ])
    demand_par_b = np.array([ stations.demand_par_b[idx]  ,\
                             sum(stations.demand_par_b) -\
                             stations.demand_par_b[idx] ])  
    distance_ij = np.zeros((2,2))
    distance_ij[0,0] = stations.distance_ij[idx, idx]
    distance_ij[0,1] = np.dot(np.delete(stations.prob_ij[idx], idx),\
                               np.delete(stations.distance_ij[idx], idx))
    distance_ij[1,0] = np.dot(np.delete(stations.prob_ij[:,idx], idx),\
                               np.delete(stations.distance_ij[:,idx], idx))
    distance_ij[1,0] = np.dot(np.delete(stations.distance_ij, idx, axis=1).ravel(),\
                              np.delete(stations.prob_ij, idx, axis=1).ravel()) 
    distance_ij[1,1] = np.dot(np.delete(np.delete(stations.distance_ij, idx,axis=0), idx,axis=1).ravel(),\
                               np.delete(np.delete(stations.prob_ij, idx,axis=0), idx,axis=1).ravel())
    
    prob_ij = np.zeros((2,2))
    prob_ij[0,0] = stations.prob_ij[idx, idx]
    prob_ij[0,1] = 1 -prob_ij[0,0]
    prob_being_in_station_j = np.empty(stations.num_stations-1)
    prob_being_in_station_j.fill(1./stations.num_stations)
    prob_ij[1,0] = np.dot(np.delete(stations.prob_ij[:,idx], idx), prob_being_in_station_j)
   # prob_ij[1,0] = np.mean(np.delete(stations.prob_ij[:,idx], idx)) ???????????
    prob_ij[1,1] = 1 - prob_ij[1,0]
    
    epsilons_support = np.zeros(2)
    epsilons_support[0] = stations.epsilons_support[idx]
    epsilons_support[1] = sum(stations.epsilons_support) - stations.epsilons_support[idx]
    
    pmin = np.array([stations.pmin[idx], np.mean(np.delete(stations.pmin, idx))])
    lost_sales_cost = np.array([stations.lost_sales_cost[idx], np.mean(np.delete(stations.lost_sales_cost, idx))])

    return demand_par_a, demand_par_b, distance_ij, prob_ij, epsilons_support, pmin, lost_sales_cost

class DP():
    '''
    Solves the DP for the 2 stations problem
    '''
    def __init__(self,indx, num_stages = num_stages_def,\
                  discount_rate = discount_rate_def ):
        demand_par_a_def, demand_par_b_def, distance_ij_def, prob_ij_def,\
        epsilons_support_def, pmin_def, lost_sales_cost_def = create_2_stations_problem(indx)
        self.indx = indx
        self.two_stations = Stations(num_stations = 2 ,\
                         num_cars = stations.num_cars,\
                         demand_par_a = demand_par_a_def,\
                         demand_par_b = demand_par_b_def,\
                         epsilons_support = epsilons_support_def,\
                         prob_ij = prob_ij_def,\
                         distance_ij =  distance_ij_def,\
                         pmin = pmin_def,\
                         lost_sales_cost = lost_sales_cost_def
                         )
        self.actions= np.array([(i,j) for i in range(self.two_stations.dmin[0],\
                                self.two_stations.dmax[0]+1) \
                                for j in range(self.two_stations.dmin[1],\
                                               self.two_stations.dmax[1]+1)])
        self.states =np.array(range(self.two_stations.num_cars+1))        
        self.discount_rate = discount_rate
        self.num_stages = num_stages
        low = -self.two_stations.epsilons_support.astype(int)
        high = self.two_stations.epsilons_support.astype(int) + 1
        self.ranges=[]
        self.probEps=1   
        for i in range(self.two_stations.num_stations):
            self.ranges.append(range(low[i],high[i]))
            self.probEps *= 1./(high[i]-low[i]) 
        self.eps_vec_1 = np.array(list(self.ranges[0]))
        self.eps_vec_2 = np.array(list(self.ranges[1]))
            
    def rang(self, x):
        '''
        TODO
        '''
        ends = np.cumsum(x)
        ranges = np.arange(ends[-1])
        rangess = ranges - np.repeat(ends-x, x)
        ww=np.repeat(x-1, x)
        return rangess, ww 

        
    def Exp_Val(self, action, state, stateValue, t):
        '''
        TODO
        '''
#        start_time = timeit.default_timer()
        demand = action
        price=self.two_stations.P(demand)   
        demand_1=demand[0] + self.eps_vec_1
        demand_2=demand[1] + self.eps_vec_2

        w1=np.minimum( demand_1, state)
        w2=np.minimum( demand_2, self.two_stations.num_cars-state)
        
        num_lost_sales_1 = demand_1 - w1
        num_lost_sales_2 = demand_2 - w2
        
        # TODO check !!!!!!!!!!!!!! 
        w11, w1_repeat = self.rang(w1+1)
        w21, w2_repeat = self.rang(w2+1)
        num_lost_sales_1_repeat = np.repeat(num_lost_sales_1, w1 + 1)
        num_lost_sales_2_repeat = np.repeat(num_lost_sales_2, w2 + 1)
        
        temp=w11[:, None] + w21
        temp = temp.ravel()
        b1 = ss.binom.pmf(w11, w1_repeat, self.two_stations.prob_ij[0,0])
        b2 = ss.binom.pmf(w21, w2_repeat,  self.two_stations.prob_ij[1,0])
        prob_array = b1[:, None] * b2 *self.probEps
        prob = prob_array.ravel()
        w1_repeat_w2=np.repeat(w1_repeat, len(w21))
        newState =temp + state - w1_repeat_w2 # TODO check here
        reward= (w1_repeat*price[0] - num_lost_sales_1_repeat * self.two_stations.lost_sales_cost[0])[:, None]\
                + (w2_repeat*price[1] - num_lost_sales_2_repeat * self.two_stations.lost_sales_cost[1] )  # TODO subtract lost sales cost
        reward=reward.ravel()
        ar=stateValue[t,:]
        vl=ar[newState]
#        vl=stateValue[t,newState]
        returns = sum(prob*list(reward + self.discount_rate * vl))
#        elapsed_time = timeit.default_timer() - start_time
#        print("Time="+str(elapsed_time))
        return returns   
    #### check !!!!!!!!!!!!!!
    
    def print_policy(self, t, policy):
        plt.ion()
        plt.cla()
        x = self.states
        plt.scatter(x,  [policy[t, i][0][0] for i in range(len(x))],color='k', label='d1')
        plt.scatter(x, [policy[t, i][0][1] for i in range(len(x))],color='g', label='d2')
        plt.xlabel('# of cars in first location')
        plt.ylabel('demand')
        plt.title('Demand policy of stage ' + str(t))
        plt.legend(loc='best')                    
        plt.pause(0.0001)
        plt.ioff()
        plt.show()  
        
    def print_value(self, t, value):    
        plt.ion()
        plt.cla()
        x = self.states
        plt.scatter(x,  [value[t, i] for i in range(len(x))], label='value function')
        plt.xlabel('# of cars in first location')
        plt.ylabel('value')
        plt.title('value of stage ' + str(t))
        plt.legend(loc='best')                    
        plt.pause(0.0001)
        plt.ioff()
        plt.show()       
        
    def solve(self, value, policy):  
        '''
        TODO
        '''  
        for t in range(self.num_stages - 2, -1, -1):
            print("t="+str(t))
            start_time = timeit.default_timer()
            for state in self.states:
                value_action = {}
                #### exploit monotonicity and bounded sensitivity
                if state == 0:
                    actions =  self.actions
                else: 
                    dmin = self.two_stations.dmin
                    dmax = self.two_stations.dmax
                    d1 = policy[t, state -1][0][0]
                    d2 = policy[t, state -1][0][1]
                    d1a = min(d1 + 1, dmax[0])
                    d2a = max(d2 - 1, dmin[1])
                    actions = np.array([ [d1, d2], [d1, d2a], [d1a, d2], [d1a, d2a] ])
                #### exploit monotonicity and bounded sensitivity
                for indx,action in enumerate(actions):
                    value_action[indx] = self.Exp_Val(action, state, value, t+1)
#                    print(value_action)
                value[t, state] = max(value_action.values())
                policy[t, state] =  [x for i,x in enumerate(actions) \
                                          if np.sum(np.abs(value_action[i] - value[t,state])) < 1e-9]
            elapsed_time = timeit.default_timer() - start_time
            print("Time="+str(elapsed_time))
#            print policy
            self.print_policy(t, policy)
            self.print_value(t, value)
        return value, policy 
                    
    def save_solution(self):
        '''
        TODO
        '''         
        value = np.zeros((self.num_stages, self.two_stations.num_cars+1))
        policy = np.empty((self.num_stages, self.two_stations.num_cars+1), dtype=object)#np.zeros((env.num_stages, env.MAX_CARS+1))
        start_time = timeit.default_timer()
        self.value, self.policy = self.solve(value, policy)
        elapsed_time = timeit.default_timer() - start_time
        print("Time="+str(elapsed_time))
        
        filename = 'DP_two_stations_'+str(self.indx)
        with open(str(filename)+".pkl", 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([self.two_stations, self.discount_rate, self.num_stages,\
                     self.value, self.policy, elapsed_time], f,protocol=2)
    
def create_multi_station_policy():
    '''
    TODO
    '''
    multi_station_policy=np.zeros((stations.num_stations,num_stages_def,stations.num_cars+1),dtype=object)
    for st in range(stations.num_stations):
        file_name='DP_two_stations_'+str(st)
        print(file_name)
        with open(str(file_name)+".pkl", 'rb') as f:  # Python 3: open(..., 'rb')
            two_stations, discount_rate, num_stages, value, policy, elapsed_time = pickle.load(f)            
        for t in range(num_stages_def):
            for state in range(two_stations.num_cars +1):
                if policy[t,state]== None: 
                    var1=0 
                    #var2=0
                else: 
                    var1=two_stations.P(policy[t,state][0])[0]
                    #var2=policy[t,state][0][1]
                                
                print("(station,t,state)="+str(st)+","+str(t)+","+str(state)+" p="+str(var1))
                multi_station_policy[st,t,state]= var1  
    filename = 'multi_station_policy'
    with open(str(filename)+".pkl", 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(multi_station_policy, f, protocol=2)
                
def run():
    '''
    TODO
    '''
    for i in range(stations.num_stations):
        dp=DP(i)
        print('Station'+str(i))
        print("demand_par_a=", dp.two_stations.demand_par_a)
        print("demand_par_b=", dp.two_stations.demand_par_b)
        
        print("dmin=", dp.two_stations.dmin)
        print("dmax=", dp.two_stations.dmax)
        
        dp.save_solution()
        
    create_multi_station_policy()

run()
