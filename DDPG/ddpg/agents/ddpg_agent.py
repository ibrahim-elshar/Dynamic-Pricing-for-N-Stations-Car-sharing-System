import keras.backend as K
import numpy as np
import sys
if sys.version_info[0] < 3:
    #raise Exception("Must be using Python 3")
    from agent import Agent
else:
    from .agent import Agent


from ddpg.actor import Actor
from ddpg.critic import Critic
from ddpg.memory import ReplayBuffer
from ddpg.process import OrnsteinUhlenbeck


class DDPGAgent(Agent):

    def __init__(self, actor_model, tgt_actor_model, critic_model, tgt_critic_model, action_limits,
                 actor_lr=1e-4, critic_lr=1e-3, critic_decay=1e-2, process=None, rb_size=1e6,
                 minibatch_size=64, tau=1e-3, gamma=0.99, warmup_episodes=None, logging=True):
        super(DDPGAgent, self).__init__(warmup_episodes, logging)

        self.actor = Actor(actor_model, critic_model, lr=actor_lr)
        self.tgt_actor = Actor(tgt_actor_model, tgt_critic_model, lr=actor_lr)
        self.tgt_actor.set_weights(self.actor.get_weights())

        self.critic = Critic(critic_model, lr=critic_lr, decay=critic_decay)
        self.tgt_critic = Critic(tgt_critic_model, lr=critic_lr, decay=critic_decay)
        self.tgt_critic.set_weights(self.critic.get_weights())

        self.action_limits = action_limits
        self.process = process
        self.buffer = ReplayBuffer(rb_size)
        self.minibatch_size = minibatch_size
        self.tau = tau
        self.gamma = gamma

        self.state_space = K.int_shape(critic_model.inputs[0])[1]
        self.action_space = K.int_shape(critic_model.inputs[1])[1]
        if process is None:
            self.process = OrnsteinUhlenbeck(x0=np.zeros(self.action_space), theta=0.15, mu=0,
                                             sigma=0.2)
        else:
            self.process = process

    def sense(self, s, a, r, s_new):
#        print(self.state_space)
        s = np.reshape(s, [-1, self.state_space])
        s_new = np.reshape(s_new, [-1, self.state_space])
        self.buffer.add((s, a, r, s_new))

    def act(self, s):
        s = np.reshape(s, [-1, self.state_space])
        a = self.actor(s) #self.tgt_actor(s)     # why acting with the target_ctor instead of the actor?
        # Cache.
        self.last_state = np.copy(s)
        self.last_action = np.copy(a)
        if self.learning_phase:
            a = self.process(a)
            #a += self.process() # for OU process
        a = np.clip(a, self.action_limits[0], self.action_limits[1])

        self.last_action_noisy = np.copy(a)
        return a[0]

    def train_step(self):
        minibatch = self.buffer.sample(self.minibatch_size)
        states = np.zeros([len(minibatch), self.state_space])
        states_new = np.zeros([len(minibatch), self.state_space])
        actions = np.zeros([len(minibatch), self.action_space])
        r = np.zeros([len(minibatch), 1])

        for i in range(len(minibatch)):
            states[i], actions[i], r[i], states_new[i] = minibatch[i]

        critic_out = self.critic(states_new, self.actor(states_new))
        tgt_critic_out = self.tgt_critic(states_new, self.tgt_actor(states_new))
        if self.logging:
            log = [('s', self.last_state),
                   ('a', self.last_action),
                   ('a_noisy', self.last_action_noisy),
                   ('q', self.critic(self.last_state, self.last_action)),
                   ('q_tgt', self.tgt_critic(self.last_state, self.last_action)),
                   (('mse', np.mean(np.square(critic_out - tgt_critic_out))))]
            self.add_log(log)

        ys = r + self.gamma * tgt_critic_out
        loss = self.critic.step(states, actions, ys)  # update critic by minimizing the loss
        self.actor.step(states)                # update actor using the sampled policy gradient

        # Soft weight update. (update the target networks)
        critic_weights = self.critic.get_weights()
        tgt_critic_weights = self.tgt_critic.get_weights()
        actor_weights = self.actor.get_weights()
        tgt_actor_weights = self.tgt_actor.get_weights()

        for i in range(len(critic_weights)):
            tgt_critic_weights[i] = (1 - self.tau) * tgt_critic_weights[i] + \
                self.tau * critic_weights[i]
        self.tgt_critic.set_weights(tgt_critic_weights)

        for i in range(len(actor_weights)):
            tgt_actor_weights[i] = (1 - self.tau) * tgt_actor_weights[i] + \
                self.tau * actor_weights[i]
        self.tgt_actor.set_weights(tgt_actor_weights)
        
        return loss

    def new_episode(self):
        self.process.clear()
        if self.logging:
            self.logs.append({})
            if len(self.logs) == 1:
                self.logs[-1]['episode'] = 1  # Initial episode.
            else:
                self.logs[-1]['episode'] = self.logs[-2]['episode'] + 1
    
    def save_weights(self, actor_suffix, critic_suffix):
        self.actor.save_model_weights(actor_suffix)
        self.critic.save_model_weights(critic_suffix)
        