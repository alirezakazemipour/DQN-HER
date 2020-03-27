from torch import nn
from torch.nn import functional as F
import torch


class DQN(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(DQN, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.hidden = nn.Linear(self.n_inputs, 256)
        nn.init.kaiming_normal_(self.hidden.weight)
        self.hidden.bias.data.zero_()

        self.output = nn.Linear(256, self.n_outputs)
        nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.zero_()

    def forward(self, states, goals):
        x = torch.cat([states, goals], dim=-1)
        x = F.relu(self.hidden(x))
        return self.output(x)
