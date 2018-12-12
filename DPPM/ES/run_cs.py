# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 22:01:05 2018

@author: Ibrahim
"""
import matplotlib.pyplot as plt
import timeit
from  feed_forward_network import FeedForwardNetwork
#from cs_baseline import *
from cs import *

ENV_NAME = 'AdpCarsharing-v0'
train_for = 5000
num_eps_tr_per_curve =50
layer_sizes=[5,20,5]
pop_size = 50
sigma =  0.03
learning_rate = 0.003
print("alpha/(n*sigma)=", learning_rate/(pop_size*sigma))
EPS_AVG = 1
decay = 1
num_threads= 1
weights_filename='weights/weights_'
GAMMA = 1

agent = Agent(GAMMA, ENV_NAME, FeedForwardNetwork(layer_sizes), pop_size,\
              sigma, learning_rate,EPS_AVG, decay, num_threads, weights_filename)

# the pre-trained weights are saved into 'weights.pkl' which you can use.

#t=agent.load_mod('weights/weights_array3_input_DP_policy_1000.pkl')
#agent.load('weights_array3_input_DP_policy_100.pkl')

#print(t)
#agent.load('weights/weights_array0_input_xt_100000.pkl')

#optimized_weights = agent.es.get_weights()
#agent.model.set_weights(optimized_weights)

## play one episode
#episodes_return_xt=agent.play(100)
#episodes_return_MPDP_policy_5000=agent.play(100)
#episodes_return_MPDP_policy_100000=agent.play(100)


start_time = timeit.default_timer()
## train for 100 iterations
agent.train(train_for, num_eps_tr_per_curve)
agent.save(weights_filename+str(train_for)+'.pkl')

elapsed_time = timeit.default_timer() - start_time
print("time="+str(elapsed_time))
#time to train 50000 on laptop = 3447.89485908 sec pop_size=20

#t=agent.load_mod('weights/weights_array5_input_DP_policy_1L_20NN_DF1_tanh_15000.pkl') #weights_array3_input_DP_policy_1L_20NN_DF1_tanh_90000
#print(t)
def run():
    episodes_return_MPDP_policy=agent.play(1000)
    mean_return_DP, episodes_return_DP_policy =  agent.agent_DP_policy(1000)
    mean_return_static, episodes_return_static_policy =  agent.agent_STATIC_policy(1000)
    mean_return_MPDP,episodes_return_MPDP_policy=agent.agent_MPDP_policy(1000)
    
    plt.figure()
    print(episodes_return_MPDP_policy)
    plt.plot(episodes_return_DP_policy,label='DP')
    plt.plot(episodes_return_static_policy,label='static')
    plt.plot(episodes_return_MPDP_policy,label='PMDP')
    
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.legend(loc='best')
    plt.show()
    
    print("mean_return_DP="+str(mean_return_DP))
    print("mean_return_static="+str(mean_return_static))
    print("mean_return_MPDP="+str(mean_return_MPDP))
    print("diff="+str(mean_return_DP-mean_return_MPDP))
    
run()