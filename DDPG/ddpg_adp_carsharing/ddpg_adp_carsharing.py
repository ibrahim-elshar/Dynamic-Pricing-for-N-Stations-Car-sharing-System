from __future__ import print_function 
import env
import gym
import adp_gym_carsharing
#import keras.backend as K
from keras.initializers import RandomUniform, VarianceScaling, normal
import keras.layers
from keras.layers import Input, Dense, concatenate, Lambda, add, BatchNormalization
from keras.models import Model
#from keras import optimizers
#from ddpg import actor, critic, process
from ddpg.agents import ddpg_agent
from ddpg.process import OrnsteinUhlenbeck, normal_noise
import numpy as np

ENV_NAME = 'AdpCarsharing-v0'
BUFFER_SIZE = 1000000
BATCH_SIZE = 32
GAMMA = 1
TAU = 1e-3     #Target Network HyperParameters
LRA = 1e-3    #Learning rate for Actor
LRC = 1e-3     #Lerning rate for Critic
CRITIC_DECAY = 1e-2

WARMUP_EPISODES = 100
LOGGING = False

OU_theta = 0.15
OU_mu = 0
OU_sigma = 0.2
######################
action_noise = 2.0
action_noise_decay = 0.9995
#TODO add layers size as input for both critic and actor models
#TODO add the stage t as input with the state to feed to the neural network
# Actor Layers
A_LAYER1 = 32
A_LAYER2 = 32
# Critic Layers
C_LAYER1 = 32
C_LAYER2 = 32


NUM_EPISODES = 5000
MAX_STEPS = 15
RENDER_PERIOD = None


def create_actor(n_states, n_actions, action_low, action_high):
    def mapping_to_target_range( x, target_min=action_low, target_max=action_high ) :
        x02 = x + 1 # x in range(0,2)
        scale = ( target_max-target_min )/2.
        return  x02 * scale + target_min
    state_input = Input(shape=(n_states,))
    l1 = Dense(A_LAYER1, activation='linear')(state_input)
    stbn = BatchNormalization()(l1)
    w_init = VarianceScaling(scale=1./3, mode='fan_in', distribution='uniform')
    h1 = Dense(A_LAYER1, kernel_initializer=w_init,
               bias_initializer=w_init, activation='relu')(stbn)
    h2 = Dense(A_LAYER2, kernel_initializer=w_init,
               bias_initializer=w_init, activation='relu')(h1)
    w_init = RandomUniform(-3e-3, 3e-3)
    out = Dense(n_actions, kernel_initializer=w_init,
                bias_initializer=w_init, activation='tanh')(h2)
    out = Lambda(mapping_to_target_range, output_shape=(1,))(out)
#    state_input = Input(shape=(n_states,))
#    h1 = Dense(A_LAYER1, activation='relu')(state_input)
#    h2 = Dense(A_LAYER2, activation='relu')(h1)
#    out = Dense(n_actions, activation='tanh')(h2)
#    out = Lambda(mapping_to_target_range, output_shape=(1,))(out)    

#    out = Lambda(lambda x: 2 * x, output_shape=(1,))(out)  # Since the output range is -2 to 2.

    return Model(inputs=[state_input], outputs=[out])
#    S = Input(shape=[n_states])   
#    h0 = Dense(A_LAYER1, activation='relu')(S)
#    h1 = Dense(A_LAYER2, activation='relu')(h0)
#    h2 = Dense(n_actions, activation='tanh')(h1)
#    V = Lambda(mapping_to_target_range, output_shape=(1,))(h2)      
#    return Model(inputs=[S],outputs=[V])


def create_critic(n_states, n_actions):
    state_input = Input(shape=(n_states,))
    action_input = Input(shape=(n_actions,))
    l1 = Dense(C_LAYER1, activation='linear')(state_input)
    l2 = Dense(C_LAYER1, activation='linear')(action_input)
    acbn = BatchNormalization()(l2) 
    stbn = BatchNormalization()(l1)
    w_init = VarianceScaling(scale=1./3, mode='fan_in', distribution='uniform')
    h1 = Dense(C_LAYER1, kernel_initializer=w_init,
               bias_initializer=w_init, activation='relu')(stbn)
    x = concatenate([h1, acbn])
    h2 = Dense(C_LAYER2, kernel_initializer=w_init, bias_initializer=w_init, activation='relu')(x)
    w_init = RandomUniform(-3e-3, 3e-3)
    out = Dense(1, kernel_initializer=w_init, bias_initializer=w_init, activation='linear')(h2)
#    h1 = Dense(C_LAYER1,  activation='relu')(state_input)
#    x = concatenate([h1, action_input])
#    h2 = Dense(C_LAYER2, activation='relu')(x)
#    out = Dense(1, activation='linear')(h2)    
    return Model(inputs=[state_input, action_input], outputs=out)
#    S = Input(shape=[n_states])  
#    A = Input(shape=[n_actions],name='action2')   
#    w1 = Dense(C_LAYER1, activation='relu')(S)
#    a1 = Dense(C_LAYER2, activation='linear')(A) 
#    h1 = Dense(C_LAYER2, activation='linear')(w1)
#    h2 =  add([h1,a1])    #concatenate([h1, a1])#
#    h3 = Dense(C_LAYER2, activation='relu')(h2)
#    V = Dense(n_actions,activation='linear')(h3)   
#    return Model(inputs=[S,A],outputs=V)




if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    env.seed(0)
    n_actions = env.action_space.shape[0]
    n_states = env.observation_space.shape[0] + 1 # x and t as state
    n_episodes = NUM_EPISODES
    n_steps = MAX_STEPS
    render_period = RENDER_PERIOD
    PROCESS = normal_noise(action_noise, action_noise_decay)#OrnsteinUhlenbeck(x0=np.zeros(n_actions), theta=OU_theta, mu=OU_mu,
                       #                      sigma=OU_sigma)

    actor, tgt_actor = create_actor(n_states, n_actions,env.action_space.low, env.action_space.high),\
                        create_actor(n_states, n_actions,env.action_space.low, env.action_space.high )
    critic, tgt_critic = create_critic(n_states, n_actions), create_critic(n_states, n_actions)

    action_limits = [env.action_space.low, env.action_space.high]

    agent = ddpg_agent.DDPGAgent(actor, tgt_actor, critic, tgt_critic, action_limits,
                 actor_lr=LRA, critic_lr=LRC, critic_decay=CRITIC_DECAY, process=PROCESS, rb_size=BUFFER_SIZE,
                 minibatch_size=BUFFER_SIZE, tau=TAU, gamma=GAMMA, warmup_episodes=WARMUP_EPISODES, logging=LOGGING)

    agent.train(env, n_episodes, n_steps, render_period)

#    print ("Storing the logs...")
#    agent.dump_logs("logs.pkl")

    print ("Performing 5 evaluation steps...")
    agent.eval(env, 5)

    print ("Saving the model...")
    actor.save("actor.model", overwrite=True)
    critic.save("critic.model", overwrite=True)

#    import pickle
#    with open('DDPG_results_5'+".pkl", 'wb') as f:  # Python 3: open(..., 'wb')
#        pickle.dump([agent.mean_performance_episodes_return, agent.std_performance_episodes_return, agent.avg_training_episodes_return], f,protocol=2)