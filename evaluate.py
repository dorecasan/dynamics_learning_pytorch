from tqdm import tqdm
import torch
from metrics import compute_r2_score, compute_rmse, compute_mse
import numpy as np

def evaluate_per_sample(sample,config,model,loss):
    X = sample['x']
    y = sample['y']

    X, y = X.to(config['device']), y.to(config['device'])

    y_pred = model(X)
    evaluate_loss = loss(y,y_pred)

    mse = compute_mse(y, y_pred)
    rmse = compute_rmse(y, y_pred)
    r2_score = compute_r2_score(y,y_pred)


    return evaluate_loss, rmse, mse, r2_score

def evaluate_per_sample_sequence(sample, config, model, loss):
    X = sample['u']
    x0 = sample['x0']
    y = sample['y']
    t = sample['time']

    X, y, x0, t = X.to(config['device']), y.to(config['device']), x0.to(config['device']), t.to(config['device'])

    y_pred_list = []
    for i in range(config['horizon']):
      y_pred_list.append(model(X,x0,t[:,i:i+1]))

    y_pred = torch.concatenate(y_pred_list, axis = 1)
    evaluate_loss = loss(y,y_pred)


    mse = compute_mse(y, y_pred)
    rmse = compute_rmse(y, y_pred)
    r2_score = compute_r2_score(y,y_pred)


    return evaluate_loss, rmse, mse, r2_score


def evaluate_per_epoch(config, valid_loader,model, loss):
    model.eval()
    r2_score_list = []
    rmse_loss_list = []
    mse_loss_list = []
    epoch_loss = 0
    with torch.no_grad():
        for sample in tqdm(valid_loader, desc="Evaluating", leave=False):
            evaluate_loss, rmse_loss, mse_loss, r2_score = \
            evaluate_per_sample(sample, config, model, loss) if not config['sequence'] \
            else evaluate_per_sample_sequence(sample, config, model, loss)

            epoch_loss += evaluate_loss.item()
            r2_score_list.append(r2_score.detach().cpu().numpy())
            mse_loss_list.append(mse_loss.detach().cpu().numpy())
            rmse_loss_list.append(rmse_loss.detach().cpu().numpy())

    return epoch_loss/len(valid_loader), np.mean(r2_score_list), np.mean(mse_loss_list), np.mean(rmse_loss_list)

