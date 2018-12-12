import random
try:
    import _pickle as pickle
except ImportError:
    import cPickle as pickle
import numpy as np
from evolution_strategy import EvolutionStrategy
from model import Model
import gym
import gym_carsharing

###########################################
#file_name=("C:\\Users\\ije8\\Box Sync\\Pittsburgh\\Courses\\Research\\Dr. Jiang and Dr. Bidkhori\\CODE\\Github\\Carsharing\\saved_results\\array3")
#file_name=("/Users/ibrahim/Box Sync/Pittsburgh/Courses/Research/Dr. Jiang and Dr. Bidkhori/CODE/Github/Carsharing/saved_results/array3")
file_name=("array3")
with open(str(file_name)+".pkl", 'rb') as f:  # Python 3: open(..., 'rb')
        array = pickle.load(f)
###########################################
def return_individual_policy_for_each_component_in_x(x,t,array):
    x=[int(i) for i in x]
    var=[]
#    print(var)
    for i in range(len(x)):
        var.append(array[i, t, x[i]])#(7,round(x[i]/2)))#array[i, t, x[i]])#(7,round(x[i]/2)))#array[i, t, x[i]])
    flat_list=[item for sublist in var for item in sublist]
    var1=[]
    var2=[]
    for i in range(int(len(flat_list)/2)):
        var1.append(flat_list[2*i])
        var2.append(flat_list[2*i+1])
    varr=var1
    varr.extend(var2)
    ######
    ############
    varr=np.asarray(varr)
    return varr
def return_x_t_individual_policy_for_each_component_in_x(x,t,array):
    x=[int(i) for i in x]
    var=[]
#    print(var)
    for i in range(len(x)):
        var.append(array[i, t, x[i]])#(7,round(x[i]/2)))#array[i, t, x[i]])#(7,round(x[i]/2)))#array[i, t, x[i]])
    flat_list=[item for sublist in var for item in sublist]
    var1=[]
    var2=[]
    for i in range(int(len(flat_list)/2)):
        var1.append(flat_list[2*i])
        var2.append(flat_list[2*i+1])
    varr=var1
    varr.extend(var2)
    ######
    vv= [t] + x
    varr= vv + varr
    ############
    varr=np.asarray(varr)
    return varr

def return_static_policy(x,t,fixed_demand):
    x=[int(i) for i in x]
    var=[]
    for i in range(len(x)):
        var.append((fixed_demand,round(x[i]/2)))
    flat_list=[item for sublist in var for item in sublist]
    var1=[]
    var2=[]
    for i in range(int(len(flat_list)/2)):
        var1.append(flat_list[2*i])
        var2.append(flat_list[2*i+1])
    varr=var1
    varr.extend(var2)
    varr=np.asarray(varr)
    return varr
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
    seed=0

    def __init__(self,model, POPULATION_SIZE=50, SIGMA=0.05, LEARNING_RATE=0.01,EPS_AVG = 5,decay=0.9999,num_threads=-1, weights_filename='weights/weights'):
        self.env = gym.make('Carsharing-v0')
        self.model = model
        self.POPULATION_SIZE=POPULATION_SIZE
        self.SIGMA=SIGMA
        self.LEARNING_RATE=LEARNING_RATE
        self.EPS_AVG = EPS_AVG
        self.decay = decay
        self.num_threads=num_threads
        self.es = EvolutionStrategy(self.model.get_weights(), self.get_reward,weights_filename, self.POPULATION_SIZE,self.SIGMA,self.LEARNING_RATE, self.decay, self.num_threads)
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
        gamma = self.env.discount_rate
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
                action = np.round(self.get_predicted_action(return_x_t_individual_policy_for_each_component_in_x(sequence[-1],self.env.t,array),self.env.action_L, np.concatenate((self.env.ab_dmax,self.env.observation))))

#                print("action"+str(action))
                observation, reward, done, _ = self.env.step(action)
                total_reward += discount*reward
                discount=discount*gamma
                sequence = sequence[1:]
                sequence.append(observation)
            episodes_return.append(total_reward)
            print ("total reward:", total_reward)
        return episodes_return


    def train(self, iterations):
        self.es.run(iterations, print_step=1)

    
    def get_reward(self, weights):
        total_reward = 0.0
        total_reward_dp = 0.0
        self.model.set_weights(weights)
        gamma = self.env.discount_rate
        init_seed=self.seed
        for episode in range(self.EPS_AVG):
            discount=1
            observation = self.env.reset()
            state = observation
            sequence = [observation]*self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                self.exploration = max(self.FINAL_EXPLORATION, self.exploration - self.INITIAL_EXPLORATION/self.EXPLORATION_DEC_STEPS)
                if random.random() < self.exploration:
                    action =return_x_t_individual_policy_for_each_component_in_x(observation,self.env.t,array)#self.env.action_space.sample(self.env.observation)
                else:
#                    print("action_L="+str(self.env.action_L))
#                    print("action_H="+str(np.concatenate((self.env.ab_dmax,self.env.observation))))
#                    print("sequence="+str(sequence))
#                    print("sequence="+str(sequence[-1]))
#                    print(np.concatenate((sequence[-1],[self.env.t])))
#                    action = np.round(self.get_predicted_action(np.concatenate((sequence[-1],[self.env.t])),self.env.action_L, np.concatenate((self.env.ab_dmax,self.env.observation))))
                    action = np.round(self.get_predicted_action(return_x_t_individual_policy_for_each_component_in_x(sequence[-1],self.env.t,array),self.env.action_L, np.concatenate((self.env.ab_dmax,observation))))
#                print("sequence"+str(sequence))
#                print("action"+str(action))
                np.random.seed(self.seed)
                observation, reward, done, _ = self.env.step(action)
#                print("self.observation=",self.env.observation) 
                self.seed += 1
                total_reward += discount * reward 
                discount=discount*gamma
                sequence = sequence[1:]
                sequence.append(observation)
#                print("sequence"+str(sequence))
            discount=1
            state =  self.env.reset()
            done_dp = False
            self.seed = init_seed
            while not done_dp:
                self.exploration = max(self.FINAL_EXPLORATION, self.exploration - self.INITIAL_EXPLORATION/self.EXPLORATION_DEC_STEPS)
                if random.random() < self.exploration:
                    action =return_x_t_individual_policy_for_each_component_in_x(observation,self.env.t,array)#self.env.action_space.sample(self.env.observation)
                else:
#                    print("action_L="+str(self.env.action_L))
#                    print("action_H="+str(np.concatenate((self.env.ab_dmax,self.env.observation))))
#                    print("sequence="+str(sequence))
#                    print("sequence="+str(sequence[-1]))
#                    print(np.concatenate((sequence[-1],[self.env.t])))
#                    action = np.round(self.get_predicted_action(np.concatenate((sequence[-1],[self.env.t])),self.env.action_L, np.concatenate((self.env.ab_dmax,self.env.observation))))
                    action_dp = return_individual_policy_for_each_component_in_x(state,self.env.t,array)
#                print("sequence"+str(sequence))
#                print("action"+str(action))
                np.random.seed(self.seed)
                (state, reward_dp, done_dp, _) = self.env.step(action_dp)
#                print("self.observation=",self.env.observation) 
                self.seed += 1
                total_reward_dp += discount * reward_dp
                discount=discount*gamma                
        return (total_reward - total_reward_dp)/self.EPS_AVG 

    def agent_DP_policy(self, ntimes=1):
      returns = 0
      episodes_return=[]
      gamma = self.env.discount_rate
      for n in range(0, ntimes):
        returns=0
        discount = 1
        done=False
        state = self.env.reset()
        while not done:
          (state, reward, done, info) = self.env.step(return_x_t_individual_policy_for_each_component_in_x(state,self.env.t,array))
          returns += discount*reward
          discount = discount*gamma
        episodes_return.append(returns)
      returns = sum(episodes_return)/ntimes
      return returns, episodes_return

    def agent_STATIC_policy(self, ntimes=1):
      returns = 0
      gamma = self.env.discount_rate
      episodes_return=[]
      for n in range(0, ntimes):
        returns=0
        discount = 1
        done=False
        state = self.env.reset()
        while not done:
          (state, reward, done, info) = self.env.step(return_static_policy(state,self.env.t,7))
          returns += discount*reward
          discount = discount*gamma
        episodes_return.append(returns)
      returns = sum(episodes_return)/ntimes
      return returns, episodes_return
  
    def agent_MPDP_policy(self, ntimes=1):
      returns = 0
      gamma = self.env.discount_rate
      episodes_return=[]
      for n in range(0, ntimes): 
        returns=0
        discount = 1
        done=False
        state = self.env.reset()
        while not done:
          action = np.round(self.get_predicted_action(return_x_t_individual_policy_for_each_component_in_x(state,self.env.t,array),self.env.action_L, np.concatenate((self.env.ab_dmax,self.env.observation))))
          (state, reward, done, info) = self.env.step(action)
          returns += discount*reward 
          discount = discount*gamma
        episodes_return.append(returns)
      returns = sum(episodes_return)/ntimes
      return returns, episodes_return
  
    
    def agent_MPXT_policy(self, ntimes=1):
      returns = 0
      gamma = self.env.discount_rate
      for n in range(0, ntimes):
        discount = 1
        done=False
        state = self.env.reset()
        while not done:
          action = np.round(self.get_predicted_action(np.concatenate((state,[self.env.t])),self.env.action_L, np.concatenate((self.env.ab_dmax,self.env.observation))))
          (state, reward, done, info) = self.env.step(action)
          returns += discount*reward 
          discount = discount*gamma
      returns = returns/ntimes
      return returns