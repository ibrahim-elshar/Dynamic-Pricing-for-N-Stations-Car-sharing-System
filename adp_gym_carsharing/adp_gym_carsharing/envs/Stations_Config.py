# -*- coding: utf-8 -*-
'''
This file contains all stations information.
'''
import numpy as np

num_stations_def = 5
num_cars_def = 50
prng = np.random.RandomState(2) #Pseudorandom number generator
prob_ij_def = prng.dirichlet(np.ones(num_stations_def),size=num_stations_def)
demand_par_a_def = prng.randint(15,25, size=num_stations_def).astype(float)
demand_par_b_def = prng.randint(3,8, size=num_stations_def).astype(float)
epsilons_support_def = prng.randint(5,6, size=num_stations_def)
distance_ij = prng.randint(1,20, size=(num_stations_def,num_stations_def)).astype(float)
distance_ij_def = (distance_ij + distance_ij.T)/2
pmin_def = np.ones(num_stations_def)
lost_sales_cost_def = prng.randint(3,5, size=num_stations_def).astype(float)

class Stations():
    '''
    Create stations info, including number of cars, number of stations,
    and the price-dependent-demand model for each station.
    The demand models are assumed linear in price, of the form,
    D(p) = a - bp. Here a is a one dimensional array where the first element 
    corresponds to station, 2nd element to station 2,...etc. Same goes for b.
    Epsilons are the additive demand noise. The full demand model is 
    D_t(p_t) = a - b p_t + epsilon_t. 
    '''
    def __init__(self, num_stations = num_stations_def ,\
                 num_cars = num_cars_def,\
                 demand_par_a = demand_par_a_def,\
                 demand_par_b = demand_par_b_def,\
                 epsilons_support = epsilons_support_def,\
                 prob_ij = prob_ij_def,\
                 distance_ij =  distance_ij_def,\
                 pmin = pmin_def,\
                 lost_sales_cost = lost_sales_cost_def
                 ):
        
        self.num_stations = num_stations
        self.num_cars = num_cars
        self.demand_par_a = demand_par_a
        self.demand_par_b = demand_par_b
        self.epsilons_support = epsilons_support
        self.prob_ij = prob_ij
        self.distance_ij = distance_ij
        self.pmin = pmin
        self.pmax = (self.demand_par_a - self.epsilons_support )/ self.demand_par_b
        self.dmin = self.D(self.pmax)
        self.dmax = self.D(self.pmin)
        self.lost_sales_cost = lost_sales_cost
    def D(self, p):
        '''
        Deterministic demand function: returns a demand vector coresponding
        to the demand of each station for the given vector price input
        '''
        d= np.rint(self.demand_par_a - self.demand_par_b*p).astype(int)
#        d= self.demand_par_a - self.demand_par_b*p
        return d 
 
    def P(self, d):
        '''
        Returns a price vector of each station for the given vector demand input
        '''
        p=(self.demand_par_a - d)/self.demand_par_b
        return p          
  
    
