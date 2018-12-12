import random
import env
try:
    import _pickle as pickle
except ImportError:
    import cPickle as pickle
import numpy as np
from evolution_strategy import EvolutionStrategy
from model import Model
import gym
import adp_gym_carsharing


###########################################
file_name= 'E:/Box Sync/Pittsburgh/Courses/Fall 2018/IE 3186 ADP/Project/Code/Evolution_Strategies/DP/multi_station_policy'
with open(str(file_name)+".pkl", 'rb') as f:  # Python 3: open(..., 'rb')
        multi_station_policy = pickle.load(f)
###########################################
def return_DP_policy(x,t,multi_station_policy):
    x=[int(i) for i in x]
    var=[]
    for i in range(len(x)):
        var.append(multi_station_policy[i, t, x[i]])#(7,round(x[i]/2)))#multi_station_policy[i, t, x[i]])#(7,round(x[i]/2)))#multi_station_policy[i, t, x[i]])
#    flat_list=[item for sublist in var for item in sublist]
#    var1=[]
#    var2=[]
#    for i in range(int(len(flat_list)/2)):
#        var1.append(flat_list[2*i])
#        var2.append(flat_list[2*i+1])
#    varr=var1
#    varr.extend(var2)
#    varr=np.asarray(varr)
    var=np.asarray(var)    
    return var

def return_static_policy(x,t,fixed_price):
    x=[int(i) for i in x]
    var=[]
    for i in range(len(x)):
#        var.append((fixed_demand,round(x[i]/2)))
        var.append(fixed_price)
#    flat_list=[item for sublist in var for item in sublist]
#    var1=[]
#    var2=[]
#    for i in range(int(len(flat_list)/2)):
#        var1.append(flat_list[2*i])
#        var2.append(flat_list[2*i+1])
#    varr=var1
#    varr.extend(var2)
#    varr=np.asarray(varr
    var=np.asarray(var)
    return var
###########################################
#model=Model()
class Agent():

    AGENT_HISTORY_LENGTH = 1
#    POPULATION_SIZE = 100
#    EPS_AVG = 5
#    SIGMA = 0.07
#    LEARNING_RATE = 0.03
    INITIAL_EXPLORATION = 0.0
    FINAL_EXPLORATION = 0.0
    EXPLORATION_DEC_STEPS = 100000

    def __init__(self, GAMMA, ENV_NAME, model, POPULATION_SIZE=50, SIGMA=0.05,\
                 LEARNING_RATE=0.01,EPS_AVG = 5,decay=0.9999,num_threads=-1,\
                 weights_filename='weights/weights'):
        self.env = gym.make(ENV_NAME)
        self.discount_rate = GAMMA
        self.model = model
        self.POPULATION_SIZE=POPULATION_SIZE
        self.SIGMA=SIGMA
        self.LEARNING_RATE=LEARNING_RATE
        self.EPS_AVG = EPS_AVG
        self.decay = decay
        self.num_threads=num_threads
        self.es = EvolutionStrategy(self.model.get_weights(), self.get_reward,\
                                    weights_filename, self.eval_func, self.POPULATION_SIZE,\
                                    self.SIGMA,self.LEARNING_RATE, self.decay,\
                                    self.num_threads)
        self.exploration = self.INITIAL_EXPLORATION


    def get_predicted_action(self, sequence, L,H):
        prediction = self.model.predict(np.array(sequence),L ,H)
        return prediction


    def load(self, filename='weights.pkl'):
        with open(filename,'rb') as fp:
            self.model.set_weights(pickle.load(fp))
        self.es.weights = self.model.get_weights()
        
    def load_mod(self, filename='weights.pkl'):
        with open(filename,'rb') as fp:
            weights,t=pickle.load(fp)
            self.model.set_weights(weights)
            print("elapsed_time="+str(t))
        self.es.weights = self.model.get_weights()
        return t


    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.es.get_weights(), fp)


    def play(self, episodes, render=False):
        self.model.set_weights(self.es.weights)
        episodes_return=[]
        gamma = self.discount_rate
        for episode in range(episodes):
            discount=1
            total_reward = 0
            observation = self.env.reset()
            sequence = [observation]*self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                if render:
                    self.env.render()
#                action = np.round(self.get_predicted_action(np.concatenate((sequence[-1],[self.env.t])),self.env.action_L, np.concatenate((self.env.ab_dmax,self.env.observation))))
                action = self.get_predicted_action(return_DP_policy(sequence[-1],\
                                                self.env.t,multi_station_policy),\
                                                self.env.action_L, self.env.action_H)

#                print("action"+str(action))
                observation, reward, done, _ = self.env.step(action)
                total_reward += discount*reward
                discount=discount*gamma
                sequence = sequence[1:]
                sequence.append(observation)
            episodes_return.append(total_reward)
#            print ("total reward:", total_reward)
        return episodes_return


    def train(self, iterations, num_eps_tr_per_curve=50):
        self.es.run(iterations, print_step=1, num_eps_tr_per_curve=50)


    def get_reward(self, weights):
        total_reward = 0.0
        self.model.set_weights(weights)
        gamma = self.discount_rate
        for episode in range(self.EPS_AVG):
            discount=1
            observation = self.env.reset()
            sequence = [observation]*self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                self.exploration = max(self.FINAL_EXPLORATION, self.exploration - self.INITIAL_EXPLORATION/self.EXPLORATION_DEC_STEPS)
                if random.random() < self.exploration:
                    action =return_DP_policy(observation,self.env.t,multi_station_policy)#self.env.action_space.sample(self.env.observation)
                else:
#                    print("action_L="+str(self.env.action_L))
#                    print("action_H="+str(np.concatenate((self.env.ab_dmax,self.env.observation))))
#                    print("sequence="+str(sequence))
#                    print("sequence="+str(sequence[-1]))
#                    print(np.concatenate((sequence[-1],[self.env.t])))
#                    action = np.round(self.get_predicted_action(np.concatenate((sequence[-1],[self.env.t])),self.env.action_L, np.concatenate((self.env.ab_dmax,self.env.observation))))
                    action = self.get_predicted_action(return_DP_policy(sequence[-1],self.env.t,multi_station_policy),\
                                                       self.env.action_L, self.env.action_H )
#                print("sequence"+str(sequence))
#                print("action"+str(action))
                observation, reward, done, _ = self.env.step(action)
                total_reward += discount * reward 
                discount=discount*gamma
                sequence = sequence[1:]
                sequence.append(observation)
#                print("sequence"+str(sequence))
        return total_reward/self.EPS_AVG 
    
    def eval_func(self, weights, NUM_EPS ):
        returns = 0.0
        episodes_return=[]
        gamma = self.discount_rate
        self.model.set_weights(weights)
        gamma = self.discount_rate
        for episode in range(NUM_EPS):
            returns=0
            discount=1
            observation = self.env.reset()
            done = False
            while not done:
                action = self.get_predicted_action(return_DP_policy(observation,self.env.t,multi_station_policy),\
                                                       self.env.action_L, self.env.action_H )
                observation, reward, done, _ = self.env.step(action)
                returns += discount * reward 
                discount=discount*gamma
            episodes_return.append(returns)
        return np.mean(episodes_return), np.std(episodes_return)
    
    
    def agent_DP_policy(self, ntimes=1):
        returns = 0.0
        episodes_return=[]
        gamma = self.discount_rate
        for n in range(0, ntimes):
            returns=0
            discount = 1
            done=False
            state = self.env.reset()
            while not done:
                action = return_DP_policy(state,self.env.t,multi_station_policy)
                (state, reward, done, info) = self.env.step(action)
                returns += discount*reward
                discount = discount*gamma
            episodes_return.append(returns)
        mean_returns = sum(episodes_return)/ntimes
        return mean_returns, episodes_return

    def agent_STATIC_policy(self, ntimes=1):
        returns = 0
        gamma = self.discount_rate
        episodes_return=[]
        for n in range(0, ntimes):
            returns=0
            discount = 1
            done=False
            state = self.env.reset()
            while not done:
#                (state, reward, done, info) = self.env.step(return_static_policy(state,\
#                                                self.env.t, min(self.env.action_H)))
                (state, reward, done, info) = self.env.step((self.env.action_L + self.env.action_H)/2)                
                returns += discount*reward
                discount = discount*gamma
            episodes_return.append(returns)
        returns = sum(episodes_return)/ntimes
        return returns, episodes_return
  
    def agent_MPDP_policy(self, ntimes=1):
        returns = 0
        gamma = self.discount_rate
        episodes_return=[]
        for n in range(0, ntimes): 
            returns=0
            discount = 1
            done=False
            state = self.env.reset()
            while not done:
              action = self.get_predicted_action(return_DP_policy(state,self.env.t,multi_station_policy),\
                                             self.env.action_L, self.env.action_H)
              (state, reward, done, info) = self.env.step(action)
              returns += discount*reward 
              discount = discount*gamma
            episodes_return.append(returns)
        returns = sum(episodes_return)/ntimes
        return returns, episodes_return
  
    
    def agent_MPXT_policy(self, ntimes=1):
        returns = 0
        gamma = self.discount_rate
        for n in range(0, ntimes):
            discount = 1
            done=False
            state = self.env.reset()
            while not done:
                action = self.get_predicted_action(np.concatenate((state,[self.env.t])),\
                                             self.env.action_L, self.env.action_H)
                (state, reward, done, info) = self.env.step(action)
                returns += discount*reward 
                discount = discount*gamma
        returns = returns/ntimes
        return returns 