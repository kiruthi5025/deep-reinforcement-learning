import random
import numpy as np




class ReplayBuffer:
def __init__(self, capacity=100000):
self.capacity = capacity
self.buffer = []
self.pos = 0


def push(self, state, action, reward, next_state, done):
data = (state, action, reward, next_state, done)
if len(self.buffer) < self.capacity:
self.buffer.append(data)
else:
self.buffer[self.pos] = data
self.pos = (self.pos + 1) % self.capacity


def sample(self, batch_size):
batch = random.sample(self.buffer, batch_size)
s, a, r, ns, d = map(np.array, zip(*batch))
return s, a, r, ns, d


def __len__(self):
return len(self.buffer)
