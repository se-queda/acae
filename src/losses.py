import torch
import torch.nn as nn
import torch.nn.functional as F


def discriminator_loss(d_pos, d_neg, alpha, beta):
    target_pos = torch.cat([torch.ones_like(alpha), alpha], dim=1) 
    target_neg = torch.cat([torch.zeros_like(beta), beta], dim=1)  

    loss_pos = F.mse_loss(d_pos, target_pos)
    loss_neg = F.mse_loss(d_neg, target_neg)    
    return loss_pos + loss_neg


def encoder_loss(d_pos, d_neg, beta):

    fake_alpha = torch.ones_like(d_pos[:, 1:])  # shape: (batch, 1)
    target_pos = torch.cat([torch.ones_like(fake_alpha), fake_alpha], dim=1)  
    target_neg = torch.cat([torch.zeros_like(beta), beta], dim=1)             

    loss_pos = F.mse_loss(d_pos, target_pos)
    loss_neg = F.mse_loss(d_neg, target_neg)    

    return loss_pos + loss_neg
