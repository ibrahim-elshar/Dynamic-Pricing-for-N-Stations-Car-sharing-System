from __future__ import print_function
import numpy as np
import multiprocessing as mp
import timeit
import matplotlib.pyplot as plt


try:
    import _pickle as pickle
except ImportError:
    import cPickle as pickle

np.random.seed(0)


def worker_process(arg):
    get_reward_func, weights = arg
    return get_reward_func(weights)


class EvolutionStrategy(object):
    def __init__(self, weights, get_reward_func, weights_filename, eval_func, population_size=50, sigma=0.1, learning_rate=0.03, decay=0.9999999999,
                 num_threads=1):

        self.weights = weights
        self.get_reward = get_reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.learning_rate = learning_rate
        self.decay = decay
        self.num_threads = mp.cpu_count() if num_threads == -1 else num_threads
        self.weights_filename = weights_filename
        self.eval = eval_func
        self.mean_training_episodes_return = []
        self.mean_performance_episodes_return = []
        self.std_performance_episodes_return = []

    def _get_weights_try(self, w, p):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA * i
            weights_try.append(w[index] + jittered)
        return weights_try

    def get_weights(self):
        return self.weights

    def _get_population(self):
        population = []
        for i in range(self.POPULATION_SIZE):
            x = []
            for w in self.weights:
                x.append(np.random.randn(*w.shape))
            population.append(x)
        return population

    def _get_rewards(self, pool, population):
        if pool is not None:
            worker_args = ((self.get_reward, self._get_weights_try(self.weights, p)) for p in population)
            rewards = pool.map(worker_process, worker_args)

        else:
            rewards = []
            for p in population:
                weights_try = self._get_weights_try(self.weights, p)
                rewards.append(self.get_reward(weights_try))
        rewards = np.array(rewards)
        return rewards

    def _update_weights(self, rewards, population):
        std = rewards.std()
        if std == 0:
            return
        rewards = (rewards - rewards.mean()) / std
        for index, w in enumerate(self.weights):
            layer_population = np.array([p[index] for p in population])
            update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)
            self.weights[index] = w + update_factor * np.dot(layer_population.T, rewards).T
        self.learning_rate *= self.decay

    def run(self, iterations, print_step=10, num_eps_tr_per_curve=50):
        mar=None
        episodes_return = []
        episodes_axes = []
        pc_index = 0
        plt.ion()
        start_time = timeit.default_timer()
        pool = mp.Pool(self.num_threads) if self.num_threads > 1 else None
        for iteration in range(iterations+1):

            population = self._get_population()
            rewards = self._get_rewards(pool, population)

            self._update_weights(rewards, population)

            if (iteration + 1) % print_step == 0:
                net_r=self.get_reward(self.weights)
                episodes_return.append(net_r)
                mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r       # moving average reward
                print('iter %d. reward: %f. MA: %f' % (iteration + 1, net_r, mar))
            if (iteration) % num_eps_tr_per_curve == 0:
                episodes_axes.append(pc_index)
                self.run_plots( self.weights, episodes_return, num_eps_tr_per_curve, episodes_axes)
                episodes_return = []
                pc_index += 1
                
            if (iteration + 1) % 5000 == 0: 
                elapsed_time = timeit.default_timer() - start_time
                filename=self.weights_filename+str(iteration+1)+'.pkl'
#                print("filename="+str(filename))
                with open(filename, 'wb') as fp:
                    pickle.dump([self.weights,elapsed_time], fp)

        if pool is not None:
            pool.close()
            pool.join()
        plt.ioff()
        plt.show()

    def run_plots(self, weights, episodes_return, num_eps_tr_per_curve, episodes_axes):
        ################################### training curve
        self.mean_training_episodes_return.append(np.mean(episodes_return))
        plt.cla()
        plt.plot(self.mean_training_episodes_return,label='Training curve')
        plt.xlabel('Training Epochs (1 epoch corresponds to '+str(num_eps_tr_per_curve) + ' episodes)')
        plt.ylabel('Average Reward per Episode')
        plt.title("Training curve: " +"DP Policy mixing - ES" + "-" + "Carsharing Envinronment")
        plt.legend(loc='best')                    
        plt.pause(0.0001)
        ################################### training curve
        ################################### performance curve    
        mean , std = self.eval( weights, 20)
        self.mean_performance_episodes_return.append(mean)
        self.std_performance_episodes_return.append(std)
        plt.cla()
        plt.errorbar(episodes_axes,self.mean_performance_episodes_return,self.std_performance_episodes_return,capsize=3,label='Performance curve')
        plt.xlabel('Epochs (1 epoch corresponds to '+str(num_eps_tr_per_curve) + ' episodes)')
        plt.ylabel('Mean & Std_dev Reward per Episode')
        plt.title("Performance curve: " +"DP Policy mixing - ES" + "-" + "Carsharing Envinronment")                        
        plt.legend(loc='best')
        plt.pause(0.0001)
        ################################## performance curve         