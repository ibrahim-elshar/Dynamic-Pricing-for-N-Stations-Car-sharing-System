import numpy as np

try:
    import _pickle as pickle
except ImportError:
    import cPickle as pickle
    
def mapping_to_target_range( x, target_min, target_max ) :
    x02 = x + 1 # x in range(0,2)
    scale = ( target_max-target_min )/2.
    return  x02 * scale + target_min


def truncating( x, target_min, target_max ) :
    x=np.maximum(x,target_min)
    x=np.minimum(x, target_max)
    return x


class FeedForwardNetwork(object):
    def __init__(self, layer_sizes):
        self.weights = []
        for index in range(len(layer_sizes)-1):
            self.weights.append(np.zeros(shape=(layer_sizes[index], layer_sizes[index+1])))
#            self.weights.append(np.random.randn(layer_sizes[index], layer_sizes[index+1]))

    def predict(self,inp, action_L, action_H):
        out = np.expand_dims(inp.flatten(), 0)
        for i, layer in enumerate(self.weights):
            out = np.dot(out, layer)
#            out = np.arctan(out)
            out = np.tanh(out)
#            if i==(len(self.weights)-1): 
#                out = np.tanh(out)
#            else:
#                out = np.tanh(out) + out
        return mapping_to_target_range(out[0],action_L,action_H)

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.weights, fp)

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            self.weights = pickle.load(fp)
