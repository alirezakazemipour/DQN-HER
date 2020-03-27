from model import DQN
from torch.optim.adam import Adam
from torch.nn import MSELoss
import numpy as np
from replay_memory import Memory
import torch


class Agent:
    def __init__(self, n_bits, lr, memory_size):
        self.n_bits = n_bits
        self.lr = lr
        self.memory_size = memory_size
        self.memory = Memory(self.memory_size)

        self.device = "cpu"
        self.model = DQN(n_inputs=self.n_bits).to(self.device)
        self.target_model = DQN(n_inputs=self.n_bits).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.opt = Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = MSELoss()
        self.epsilon = 1.0
        self.epsilon_decay = 0.01

    def choose_action(self, states, goals):

        if np.random.random() < self.epsilon:
            action = np.random.randint(low=0, high=self.n_bits)
        else:
            states = torch.Tensor(states, device=self.device)
            goals = torch.Tensor(goals, device=self.device)
            action = self.model(states, goals).max(dim=-1).item()
            action = np.clip(action, 0, self.n_bits)

        return action

    def update_epsilon(self):
        self.epsilon -= self.epsilon_decay

    def store(self, state, action, reward, done, next_state, goal):
        state = torch.Tensor(state, device=self.device)
        reward = torch.Tensor([reward], device=self.device)
        action = torch.Tensor(action, device=self.device)
        next_state = torch.Tensor(next_state, device=self.device)
        done = torch.Tensor([done], device=self.device)
        goal = torch.Tensor(goal, device=self.device)
        self.memory.push(state, action, reward, done, next_state, goal)
