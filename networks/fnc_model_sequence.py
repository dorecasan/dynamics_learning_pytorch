from re import X
from matplotlib import numpy
import torch
from torch import nn

class FCNModelSequence(nn.Module):
    def __init__(self, config):
        super(FCNModelSequence, self).__init__()

        h = config['horizon']
        alpha = config['model_alpha']
        nu = config['nu']
        nx = config['nx']
        q = 8
        num_outputs= q*h

        self.branch_net = BranchNet(nu,q,config)

        self.root_net = RootNet(nx,num_outputs)

        self.trunk_net = TrunkNet(1,num_outputs)

        self.time_sigmoid = TimeLayer(h,alpha)

        self.post_branch_net = PostBranchNet(q,num_outputs,config)

    def forward(self, u, x0, t):
        u = self.branch_net(u)

        sigmoid_gate = self.time_sigmoid(t)

        u = u*sigmoid_gate

        u = self.post_branch_net(u)

        t = self.trunk_net(t)

        x0 = self.root_net(x0)

        out = x0*u*t

        out = torch.sum(out,dim=1).reshape((-1,1))

        return out


class TimeLayer(nn.Module): 

  def __init__(self, h, alpha): 
    super(TimeLayer, self).__init__() 
    self.h = h 
    self.alpha = alpha

  def forward(self, t):
      outputs = []
      for i in range(0, self.h):
          weight = self.alpha * (t - i)
          output = nn.functional.sigmoid(weight)
          outputs.append(output)

      return torch.cat(outputs, dim=1).unsqueeze(-1)


class BranchNet(nn.Module):

  def __init__(self,n,num_outputs,config):
    super(BranchNet,self).__init__()
    h = config['horizon']
    self.fc_list = nn.ModuleList([
          nn.Sequential(nn.Linear(n, 16),
          nn.ReLU(),
          nn.Linear(16, num_outputs)) for i in range(h)])

  def forward(self,u):

    u = torch.cat([fc(u[:,i:i+1, :]) for i, fc in enumerate(self.fc_list)], dim=1)

    return u


class RootNet(nn.Module):

  def __init__(self,n,num_outputs):

    super(RootNet,self).__init__()
    self.fc_list = nn.Sequential(nn.Linear(n, 16),
          nn.ReLU(),
          nn.Linear(16, num_outputs))

  def forward(self, x0):

    x0 = self.fc_list(x0)

    return x0

class TrunkNet(nn.Module):

  def __init__(self,n,num_outputs):

    super(TrunkNet,self).__init__()
    self.fc_list = nn.Sequential(nn.Linear(n, 16),
          nn.ReLU(),
          nn.Linear(16, num_outputs))

  def forward(self, t):

    t = self.fc_list(t)

    return t

class PostBranchNet(nn.Module):

  def __init__(self,n,num_outputs, config):

    super(PostBranchNet,self).__init__()
    self.fc_list = nn.Sequential(
          nn.Flatten(),
          nn.Linear(n*config['horizon'], 32),
          nn.ReLU(),
          nn.Linear(32, num_outputs))

  def forward(self, x):

    x = self.fc_list(x)

    return x














