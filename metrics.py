import torch
from sklearn.metrics import r2_score
import torch.nn.functional as F
import numpy as np

def compute_r2_score(y_true, y_pred):
    """Computes the R2 score between y_true and y_pred"""
    ss_res = torch.sum(torch.square(y_true - y_pred))
    ss_tot = torch.sum(torch.square(y_true - torch.mean(y_true)))
    r2 = 1 - ss_res / ss_tot
    return r2

def compute_mse(y_true,y_pred):
    return F.mse_loss(y_true,y_pred)

def compute_rmse(y_true,y_pred):
    return torch.sqrt(F.mse_loss(y_true,y_pred))