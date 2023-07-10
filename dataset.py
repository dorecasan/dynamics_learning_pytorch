import zipfile
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch


class TempData(Dataset):

  def __init__(self, X: np.ndarray, y: np.ndarray, transform = None) -> None:
    self.X = X
    self.y = y
    self.transform = transform
    self.len = self.X.shape[0]
  
  def __getitem__(self, index: int) -> dict:
    x_data = self.X[index].astype('float32')
    y_data = self.y[index].astype('float32')

#    if self.transform:
#      agumented = self.transform(image = x_data, mask = y_data)
#      x_data = agumented['image']
#      y_data = agumented['mask']
    x_data = torch.from_numpy(x_data)
    y_data = torch.from_numpy(y_data)

    sample = {'x': x_data,
              'y': y_data}
    
    return sample

  def __len__(self) -> int:
    return self.len



class TempDataSequence(Dataset):

  def __init__(self, X: np.ndarray, y: np.ndarray, config, transform = None) -> None:
    self.X = X
    self.y = y
    self.transform = transform
    self.len = self.X.shape[0]
    self.h = config['horizon']
  
  def __getitem__(self, index: int) -> dict:
    x_data = self.X[index].astype('float32')
    y_data = self.y[index].astype('float32')

    x0 = x_data[0:1,0]
    u = x_data[:,1:]
    t = torch.linspace(1,self.h,self.h)


#    if self.transform:
#      agumented = self.transform(image = x_data, mask = y_data)
#      x_data = agumented['image']
#      y_data = agumented['mask']
    u = torch.from_numpy(u)
    y_data = torch.from_numpy(y_data).squeeze(-1)

    sample = {'u': u,
              'x0': x0,
              'time': t,
              'y': y_data}
    
    return sample

  def __len__(self) -> int:
    return self.len
  

if __name__ == "main":
    zip_file = zipfile.ZipFile('data-learning-control-room-temperature.zip')
    zip_file.extractall(path='/content/')
    zip_file.close()
    data = pd.read_csv('/content/data-B90102-30m.csv')
    x_data = data[['room_temp','supply_temp','airflow_current']].values
    y_data = data['room_temp'].values
    y_data = y_data.reshape((-1,1))
    trainData = TempData(x_data, y_data)

    for idx in tqdm(range(len(trainData))):
      sample = trainData[idx]
      print("Shape of x: {}".format(sample['x'].shape))
      print("Shape of y: {}".format(sample['y'].shape))