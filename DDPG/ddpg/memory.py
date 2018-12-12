from collections import deque
import random


class ReplayBuffer(object):

    def __init__(self, maxlen):
        self.maxlen = int(maxlen)
        self.buffer = deque(maxlen=self.maxlen)

    def add(self, el):
        self.buffer.append(el)

    def sample(self, size):
        if size >= len(self.buffer):
            return list(self.buffer)
        return random.sample(list(self.buffer), size)

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        for i in self.buffer:
            yield i
