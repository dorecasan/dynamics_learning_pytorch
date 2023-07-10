import torch
from torch import nn


class FCNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3,16)
        self.linear2 = nn.Linear(16,32)
        self.linear3  = nn.Linear(32,1)

        self.silu1 = nn.SiLU()
        self.silu2 = nn.SiLU()

    def forward(self,x):
        x= self.linear1(x)
       # x = self.silu1(x)
        x = self.linear2(x)
       # x = self.silu2(x)
        x = self.linear3(x)

        return x

class LSTMModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.lstm1 = nn.LSTM(3,64)
    self.linear1 = nn.Linear(64,1)

  def forward(self,x):
    x,(hn,hc) = self.lstm1(x)
    x = self.linear1(x)

    return x