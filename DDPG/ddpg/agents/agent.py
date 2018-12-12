from __future__ import print_function 
from collections import deque
from timeit import default_timer as timer
try:
    import cPickle as pickle
except:
    import pickle
import matplotlib.pyplot as plt
import numpy as np

class Agent(object):

    def __init__(self, warmup_episodes, logging):
        self.learning_phase = True
        self.warmup_episodes = warmup_episodes if warmup_episodes is not None else 0
        self.logging = logging
        if logging:
            self.logs = []
            print ("Logging enabled!")
        else:
            print ("Logging disabled!")

    def sense(self, s, a, r, s_new):
        raise NotImplementedError()

    def act(self, s):
        raise NotImplementedError()

    def new_episode(self, episode):
        raise NotImplementedError()

    def train_step(self):
        raise NotImplementedError()

    def add_log(self, agent_returns):
        if not self.logging:
            raise RuntimeError('Logging disabled!')
        if not self.logs:
            self.logs.append({})
            self.logs[-1]['episode'] = 1  # Initial episode.
        for name, value in agent_returns:
            if name not in self.logs[-1]:
                self.logs[-1][name] = []
            self.logs[-1][name].append(value)

    def dump_logs(self, filename):
        if not self.logging:
            raise RuntimeError('Logging disabled!')
        pickle.dump(self.logs, open(filename, "wb"))

    def train(self, env, n_episodes, n_steps, render_period=None, reward_window=100):  
        start = timer()
        try:
            #################
            episodes_return = []
            num_episodes_to_update_training_curve = 50
            save_weights_num_episodes = 100
            self.avg_training_episodes_return = []
            self.mean_performance_episodes_return = []
            self.std_performance_episodes_return = []
            episodes_axes = []
            num_episodes_after_warm_up = 0
            pc_index = 0
            #################
            plt.ion()
            total_r = 0
            avg_ep_r_hist = []
#            avg_step_r_hist = []
#            tot_num_steps = 1
            #################            
#            rewards = deque(maxlen=reward_window)
            for episode in range(1, n_episodes + 1):
                self.new_episode()
                s = env.reset()
                s = np.append(s,[0])
#                r_sum = 0
                ##########
                ep_r = 0    
                avg_loss = []
                ##########
                for t in range(1, n_steps):
                    if render_period is not None and episode % render_period == 0:
                        env.render()
                    a = self.act(s)
                    s_new, r, done, _ = env.step(a)
                    s_new = np.append(s_new,[t])
                    self.sense(s, a, r, s_new)
                    if episode > self.warmup_episodes:
                        loss = self.train_step()
                        avg_loss.append(loss)
                        
#                        print("Episode", episode, "Step", tot_num_steps, "Action", a, "Reward", r, "Loss", loss) 
#                    else: 
#                        print("Episode", episode, "Step", tot_num_steps, "Action", a, "Reward", r) 
#                    r_sum += r
                    s = s_new
                    #########
                    ep_r += r
#                    total_r += r   
#                    tot_num_steps += 1 
                    #########
#                    if tot_num_steps >= 10: 
#                        avg_step_r = total_r/tot_num_steps
#                        avg_step_r_hist.append(avg_step_r)
#                        if tot_num_steps % 20 == 0:
#                            print('No. of steps %d Avg Reward/Step %s' % (tot_num_steps, avg_step_r))
                              
                    #########
                    if done:
                        break
                if episode > self.warmup_episodes:
                    print("Episode: ", num_episodes_after_warm_up," Total Episode Reward: ", ep_r, "AVG Loss", np.mean(avg_loss) )
                ################################### training curve
#                    if num_episodes_after_warm_up > self.warmup_episodes:
                    episodes_return.append(ep_r)
                    avg_loss= []
                    if num_episodes_after_warm_up %  num_episodes_to_update_training_curve == 0:
                        self.avg_training_episodes_return.append(np.mean(episodes_return))
                        episodes_return = []
                        plt.cla()
                        plt.plot(self.avg_training_episodes_return,label='Training curve')
                        plt.xlabel('Training Epochs (1 epoch corresponds to '+str(num_episodes_to_update_training_curve) + ' episodes)')
                        plt.ylabel('Average Reward per Episode')
                        plt.title("Training curve: " +"DDPG" + "-" + "Carsharing Envinronment")
                        plt.legend(loc='best')                    
                        plt.pause(0.0001)
                    if num_episodes_after_warm_up % save_weights_num_episodes == 0:
                        c_filename = 'Critic_DDPG_weights_'+str(num_episodes_after_warm_up)
                        a_filename = 'Actor_DDPG_weights_'+str(num_episodes_after_warm_up)
                        self.save_weights(a_filename, c_filename)
                ################################### training curve
                ################################### performance curve    
                    if  num_episodes_after_warm_up % num_episodes_to_update_training_curve == 0:
                        mean , std = self.eval(env, n_episodes=20, n_steps=200, render=False)
                        self.mean_performance_episodes_return.append(mean)
                        self.std_performance_episodes_return.append(std)
                        episodes_axes.append(pc_index)
                        plt.cla()
                        plt.errorbar(episodes_axes,self.mean_performance_episodes_return,self.std_performance_episodes_return,capsize=3,label='Performance curve')
                        plt.xlabel('Epochs (1 epoch corresponds to '+str(num_episodes_to_update_training_curve) + ' episodes)')
                        plt.ylabel('Mean & Std_dev Reward per Episode')
                        plt.title("Performance curve: " +"DDPG" + "-" + "Carsharing Envinronment")                        
                        plt.legend(loc='best')
                        plt.pause(0.0001)
                        pc_index += 1
                    num_episodes_after_warm_up += 1
#                ################################## performance curve 
#                rewards.append(float(ep_r) / t)
#                print ("Ep %4d, last episode mean reward per step %.5f, mean reward per step %.5f." % (episode, rewards[-1],
#                                                                      sum(rewards) / len(rewards)))                
#                ########################
#                plt.cla()
#                plt.plot(avg_step_r_hist, label='training_curve')
#                plt.xlabel('Training Steps')
#                plt.ylabel('Average Reward per Step')
#                plt.title("Training curve: " +"DDPG" + "-" + "Carsharing Envinronment")
#                plt.legend(loc='best')                                    
#                plt.pause(0.0001)         
#                #########################
#                if episode >= 1: 
#                    avg_ep_r = total_r/(episode)
#                    avg_ep_r_hist.append(avg_ep_r)
#                    if episode % 20 == 0:
#                        print('Episode %d Avg Reward/Ep %s' % (episode, avg_ep_r))
#                plt.cla()
#                plt.plot(avg_ep_r_hist, label='training_curve')
#                plt.xlabel('Training Episodes')
#                plt.ylabel('Average Reward per Episode')
#                plt.title("Training curve: " +"DDPG" + "-" + "Carsharing Envinronment")
#                plt.legend(loc='best')                                    
#                plt.pause(0.0001)                
            plt.ioff()
            plt.show()                
                ########################
                
        except KeyboardInterrupt:
            print ("Training interrupted by the user!")

        end = timer()
        duration = end - start
        print ("Performed %d episodes. Elapsed time %f. Average time per episode %f." % \
            (episode - 1, duration, duration / (episode - 1)))

    def eval(self, env, n_episodes=1, n_steps=200, render=False):
        learning_phase = self.learning_phase
        self.learning_phase = False
        returns = 0
        episodes_return = []
        try:
            for episode in range(n_episodes):
#                print ("Ep %4d" % (episode))
                done = False
                returns = 0
                s = env.reset()
                s = np.append(s,[0])
                for t in range(1,n_steps):
                    if render:
                        env.render()
                    a = self.act(s)
                    s_new, r, done, _ = env.step(a)
                    s_new = np.append(s_new,[t])
                    s = s_new
                    returns += r
                    if done:
                        break
                episodes_return.append(returns)
        except KeyboardInterrupt:
            print ("Evaluation interrupted by the user!")

        self.learning_phase = learning_phase
        return np.mean(episodes_return) , np.std(episodes_return)
    

