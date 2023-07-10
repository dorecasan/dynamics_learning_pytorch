import torch
from metrics import compute_mse, compute_r2_score, compute_rmse
from tqdm import tqdm
import numpy as np
from evaluate import evaluate_per_epoch
from utils.train_utils import epoch_time
import time

def train_per_sample(sample, config, model, loss, optimizer):
    X = sample['x']
    y = sample['y']

    X, y = X.to(config['device']), y.to(config['device'])

    optimizer.zero_grad()
    y_pred = model(X)
    train_loss = loss(y,y_pred)

    train_loss.backward()
    optimizer.step()

    mse = compute_mse(y, y_pred)
    rmse = compute_rmse(y, y_pred)
    r2_score = compute_r2_score(y,y_pred)


    return train_loss, rmse, mse, r2_score

def train_per_sample_sequence(sample, config, model, loss, optimizer):
    X = sample['u']
    x0 = sample['x0']
    y = sample['y']
    t = sample['time']

    X, y, x0, t = X.to(config['device']), y.to(config['device']), x0.to(config['device']), t.to(config['device'])

    optimizer.zero_grad()

    y_pred_list = []
    for i in range(config['horizon']):
      y_pred_list.append(model(X,x0,t[:,i:i+1]))

    y_pred = torch.concatenate(y_pred_list, axis = 1)
    train_loss = loss(y,y_pred)

    train_loss.backward()
    optimizer.step()

    mse = compute_mse(y, y_pred)
    rmse = compute_rmse(y, y_pred)
    r2_score = compute_r2_score(y,y_pred)


    return train_loss, rmse, mse, r2_score

def train_per_epoch(config, train_dataloader, model, loss, optimizer):
    model.train()
    epoch_loss= 0
    r2_score_list = []
    rmse_loss_list = []
    mse_loss_list = []

    for sample in tqdm(train_dataloader, desc='training', leave=False):

        train_loss, rmse_loss, mse_loss, r2_score = \
          train_per_sample(sample, config, model, loss, optimizer) if not config['sequence'] \
          else train_per_sample_sequence(sample, config, model, loss, optimizer)

        epoch_loss += train_loss.item()

        r2_score_list.append(r2_score.detach().cpu().numpy())
        mse_loss_list.append(mse_loss.detach().cpu().numpy())
        rmse_loss_list.append(rmse_loss.detach().cpu().numpy())    

    return epoch_loss / len(train_dataloader), np.mean(r2_score_list), np.mean(mse_loss_list), np.mean(rmse_loss_list)


def train_and_evaluate(config, train_dataloader, valid_dataloader, model, loss, optimizer, scheduler, start_epoch):
    best_value_loss = float('inf')
    start_time = time.monotonic()
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(start_epoch,start_epoch+ config['num_epochs']):
        train_loss, train_r2_score, train_mse_loss, train_rmse_loss = train_per_epoch(config,train_dataloader,model,loss,optimizer)
        valid_loss, valid_r2_score, valid_mse_loss, valid_rmse_loss = evaluate_per_epoch(config,valid_dataloader,model,loss)

        scheduler.step(valid_loss)

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print("----------------------------------------------------------------------------------------------")
        print(f"Epoch: {epoch+1} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print("Training")
        print(
        f"\t MSE Loss: {train_loss:.8f} | R2 score: {train_r2_score:.5f} | RMSE loss: {train_rmse_loss:.8f}")
        print("Validating")
        print(
        f"\t MSE Loss: {valid_loss:.8f} |  R2 score: {valid_r2_score:.5f} | RMSE loss: {valid_rmse_loss:.8f}")
        print("----------------------------------------------------------------------------------------------")
        print("\n")

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

    return train_loss_list, valid_loss_list






