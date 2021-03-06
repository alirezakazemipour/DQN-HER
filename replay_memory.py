import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward','done', 'next_state', 'goal'))


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, *item):
        self.memory.append(Transition(*item))
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        assert len(self.memory) <= self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
