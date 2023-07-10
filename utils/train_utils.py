from albumentations import Compose
from albumentations.pytorch import ToTensorV2
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from dataset import TempData, TempDataSequence
from networks.fnc_model import *
from networks.fnc_model_sequence import *
import scipy.io
from pathlib import Path

def apply_transform(config):
    valid_transform = Compose([
        ToTensorV2()
    ])
    train_transform = Compose([
        ToTensorV2()
    ])

    return train_transform, valid_transform

def prepare_dataloaders(config, data, train_transform, valid_transform):
  if not config['sequence']:
    train_data = TempData(data['x_train'],data['y_train'],train_transform)
    valid_data = TempData(data['x_valid'],data['y_valid'],valid_transform)
  else:
    train_data = TempDataSequence(data['x_train'],data['y_train'],config,train_transform)
    valid_data = TempDataSequence(data['x_valid'],data['y_valid'],config,valid_transform)

  train_dataloader = DataLoader(train_data, batch_size = config['batch_size'], shuffle = config['random_split'])
  valid_dataloader = DataLoader(valid_data, batch_size = config['batch_size'], shuffle = config['random_split'])

  return train_dataloader, valid_dataloader

def prepare_objectives(config, model):

    loss  = nn.MSELoss()

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr = config['learing_rate'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr = config['learning_rate'])
    elif config['optimizer'] == 'Nadam':
        optimizer = optim.NAdam(model.parameters(), lr = config['learning_rate'])

    if config['scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True)

    return loss, optimizer, scheduler

def prepare_models(config):
    model = None
    if config['model'] == 'fcn':
      model = FCNModel()
    elif config['model'] == 'lstm':
      model = LSTMModel()
    elif config['model'] == 'fcn_sequence':
      model = FCNModelSequence(config)
    return model

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def save_model(config,model):
  print("Saving model state to: ", config['save_model_checkpoint'])
  torch.save({'model_state_dict': model.state_dict(),
            }, config['save_model_checkpoint'])

def save_all_states(config,model,best_valid_loss,optimizer,epoch):
  Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
  print("Saving checkpoints ", config['save_checkpoint'])
  torch.save({'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': best_valid_loss,
              }, config['save_checkpoint'])

def save_model_to_matlab(config,model):
  model_state = model.state_dict()
  state_dict = dict(model_state.items())
  state_dict_matlab = {}
  for key, value in state_dict.items():
    key_mt = 'layer_'+'_'.join(key.split('.'))
    if isinstance(value, torch.Tensor):
        state_dict_matlab[key_mt] = value.detach().numpy()
    else:
        state_dict_matlab[key_mt] = value
  
  scipy.io.savemat(config['save_matlab_checkpoint'], mdict=state_dict_matlab)

    
