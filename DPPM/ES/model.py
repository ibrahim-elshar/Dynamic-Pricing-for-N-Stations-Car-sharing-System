import numpy as np


def mapping_to_target_range( x, target_min, target_max ) :
    x02 = x + 1 # x in range(0,2)
    scale = ( target_max-target_min )/2.
    return  x02 * scale + target_min


def truncating( x, target_min, target_max ) :
    x=np.maximum(x,target_min)
    x=np.minimum(x, target_max)
    return x



class Model(object):

    def __init__(self):
#        self.weights = [np.zeros(shape=(11, 80)), np.zeros(shape=(80, 80)), np.zeros(shape=(80, 20))] #24 is the dimension of the state. 4 is the dimension of the action       
#        self.weights = [np.zeros(shape=(20, 20)), np.zeros(shape=(20, 20)), np.zeros(shape=(20, 20))] #24 is the dimension of the state. 4 is the dimension of the action       
        self.weights = [np.random.randn(20, 40),np.random.randn(40, 40),np.random.randn(40, 20)] #24 is the dimension of the state. 4 is the dimension of the action       
#        self.weights = [np.identity(20),np.identity(20)] #24 is the dimension of the state. 4 is the dimension of the action       
#        self.weights = [np.random.randn(20, 40),np.random.randn(40, 20)] #24 is the dimension of the state. 4 is the dimension of the action       

#        self.weights = [np.random.normal(0, 1, size=(20, 80)), np.random.normal(0, 1, size=(80, 80)), np.random.normal(0, 1, size=(80, 20))]
#        self.weights = [np.zeros(shape=(20, 120)), np.zeros(shape=(120, 120)),np.zeros(shape=(120, 120)), np.zeros(shape=(120, 20))] #24 is the dimension of the state. 4 is the dimension of the action
#        self.weights = [np.random.random(size=(20, 20)), np.random.random(size=(20, 20)),np.random.random(size=(20, 20)), np.random.random(size=(20, 20))] #24 is the dimension of the state. 4 is the dimension of the action
#

    def predict(self, inp, action_L,action_H):
        out = np.expand_dims(inp.flatten(), 0)
#        out = out / np.linalg.norm(out)
        for i,layer in enumerate(self.weights):
            out = np.dot(out, layer)
            out = np.tanh(out)
#            print(out[0])
#        return truncating(out[0], action_L, action_H)
        return mapping_to_target_range(out[0], action_L, action_H)
#    def predict(self, inp):
#        out = np.expand_dims(inp.flatten(), 0)
#        for i, layer in enumerate(self.weights):
#            out = np.dot(out, layer)
#            out = np.arctan(out)
#        return out[0]

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

