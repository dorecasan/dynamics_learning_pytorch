import zipfile
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

def preprocess(config):
    zip_file = zipfile.ZipFile(config['data_path'])
    zip_file.extractall(path=os.path.dirname(config['data_path']))
    zip_file.close()

    data = pd.read_csv(os.path.dirname(config['data_path'])+'/data-B90102-30m.csv')

    x_data = data[['room_temp','supply_temp','airflow_current']].values
    x_data = np.concatenate((x_data[:-1,0:1],x_data[1:,1:]),axis=1)
    y_data = data['room_temp'].values
    y_data = y_data[1:]
    y_data = y_data.reshape((-1,1))

    X_scaler = MinMaxScaler()
    X_scaled = X_scaler.fit_transform(x_data)
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y_data)

    split_ratio = list(map(float, config['split_ratio'].split())) 

    if config['sequence']:
      X_trunc = []
      y_trunc = []
      h = config['horizon']
      for i in range(len(X_scaled)-h):
        X_trunc.append(X_scaled[i:i+h,:])
        y_trunc.append(y_scaled[i:i+h,:])
      X_scaled = np.stack(X_trunc,axis = 0)
      y_scaled = np.stack(y_trunc,axis = 0)

    if config['random_split']:
      x_train, x_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=split_ratio[-1], random_state=42)
      x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=split_ratio[1], random_state=42)
    else:
      n = X_scaled.shape[0]
      n_train = int(n*split_ratio[0])
      n_valid = int(n*(split_ratio[0]+split_ratio[1]))
      x_train = X_scaled[:n_train]
      y_train = y_scaled[:n_train]
      x_val = X_scaled[n_train:n_valid]
      y_val = y_scaled[n_train:n_valid]
      x_test = X_scaled[n_valid:]
      y_test = y_scaled[n_valid:] 

    data_processed = dict()

    data_processed['x_train'] = x_train
    data_processed['y_train'] = y_train

    data_processed['x_valid'] = x_val
    data_processed['y_valid'] = y_val
    data_processed['x_test'] = x_test
    data_processed['y_test'] = y_test

    return data_processed, X_scaler, y_scaler

def read_data(config):
    zip_file = zipfile.ZipFile(config['data_path'])
    zip_file.extractall(path=os.path.dirname(config['data_path']))
    zip_file.close()

    data = pd.read_csv(os.path.dirname(config['data_path'])+'/data-B90102-30m.csv')
    return data
