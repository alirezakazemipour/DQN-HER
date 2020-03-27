from model import DQN
from torch.optim.adam import Adam
from torch.nn import MSELoss
import numpy as np
from replay_memory import Memory, Transition
import torch


class Agent:
    def __init__(self, n_bits, lr, memory_size, batch_size, gamma):
        self.n_bits = n_bits
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = Memory(self.memory_size)

        self.device = "cpu"
        self.model = DQN(n_inputs=2 * self.n_bits).to(self.device)
        self.target_model = DQN(n_inputs=2 * self.n_bits).to(self.device)
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
            action = self.model(states, goals).max(dim=-1)[1].detach().numpy()
            action = np.clip(action, 0, self.n_bits)

        return action

    def update_epsilon(self):
        self.epsilon -= self.epsilon_decay

    def store(self, state, action, reward, done, next_state, goal):
        state = torch.Tensor(state, device=self.device)
        reward = torch.Tensor([reward], device=self.device)
        action = torch.Tensor([action], device=self.device)
        next_state = torch.Tensor(next_state, device=self.device)
        done = torch.Tensor([done], device=self.device)
        goal = torch.Tensor(goal, device=self.device)
        self.memory.push(state, action, reward, done, next_state, goal)

    def unpack_batch(self, batch):

        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).to(self.device).view(self.batch_size, self.n_bits)
        actions = torch.cat(batch.action).to(self.device)
        rewards = torch.cat(batch.reward).to(self.device)
        next_states = torch.cat(batch.next_state).to(self.device).view(self.batch_size, self.n_bits)
        dones = torch.cat(batch.done).to(self.device)
        goals = torch.cat(batch.goal).to(self.device).view(self.batch_size, self.n_bits)

        return states, actions, rewards, dones, next_states, goals

    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, dones, next_states, goals = self.unpack_batch(batch)

        with torch.no_grad():
            target_q = rewards + self.gamma * self.target_model(next_states, goals).max(-1)[0] * (1 - dones)

        q = self.model(states, goals)[actions.long()]
        loss = self.loss_fn(q, target_q.view(self.batch_size, 1))

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.soft_update_of_target_network(self.model, self.target_model)

        return loss.item()

    @staticmethod
    def soft_update_of_target_network(local_model, target_model, tau=0.05):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
