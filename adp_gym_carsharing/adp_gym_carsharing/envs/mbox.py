
import numpy as np
#import gym

#np.random.multinomial(800, np.ones(20)/20, size=1)[0]
def randomize(n, num_terms):
    if n == 1:
        a=np.zeros(num_terms).astype(int)
        i=np.random.randint(0, num_terms)
        a[i]=1
        return [a[x] for x in range(num_terms)]
    else:
        num_terms = num_terms  - 1
        a = np.random.randint(0, n, num_terms) 
        a=np.append(a,[0,n])
        a=np.sort(a)
        return [a[i+1] - a[i] for i in range(len(a) - 1)]
    
def arandomize(n, num_terms,j):
    if n == 1:
        a=np.zeros(num_terms).astype(int)
        i=np.random.randint(0, num_terms)
        a[i]=1
        ls=[a[x] for x in range(num_terms)]
        ls.insert(j,0)
        return ls
    else:
        num_terms = num_terms  - 1
        a = np.random.randint(0, n, num_terms) 
        a=np.append(a,[0,n])
        a=np.sort(a)
        ls=[a[i+1] - a[i] for i in range(len(a) - 1)]
        ls.insert(j,0)
        return ls


class MBox():
    """
    A box in R^n.
    I.e., each coordinate is bounded.

    Example usage:
    self.action_space = spaces.Box(low=-10, high=10, shape=(1,))
    """
    def __init__(self, s=None, num_terms=None, dtype=None):
        """
        Two kinds of valid input:
            Box(low=-1.0, high=1.0, shape=(3,4)) # low and high are scalars, and shape is provided
            Box(low=np.array([-1.0,-2.0]), high=np.array([2.0,4.0])) # low and high are arrays of the same shape
        """
        if dtype is float:
            self.s = float(s)
            self.num_terms = int(num_terms)
        else:
            self.s = int(s)
            self.num_terms = int(num_terms)
            
        self.shape=(self.num_terms,)
        
#        gym.Space.__init__(self, shape, dtype)

    def sample(self,state):
        return randomize(self.s, self.num_terms)
    
    def contains(self, x):
        return  (x >= 0).all() and (sum(x) <= self.s) and (len(x) == self.num_terms)
    
    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()
    
    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return "MBox" + str(self.num_terms)
    def __eq__(self, other):
        return np.allclose(self.s, other.s) and np.allclose(self.num_terms, other.num_terms)
