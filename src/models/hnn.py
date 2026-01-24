import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import grad
from typing import Tuple, Optional
import math


#input : batch, time, feature_phy projected to Dimension D = L//2
class HNN(nn.Module):
    
    def __init__(self, hnn_dim, steps, dt):
        super().__init__()
        self.feature_dim = hnn_dim
        self.steps = steps
        self.dt = dt

        self.h_dense1 = nn.Linear(2*hnn_dim, 2*hnn_dim)
        self.h_dense2 = nn.Linear(2*hnn_dim, 2*hnn_dim)
        self.out = nn.Linear(2*hnn_dim, 1, bias = False)
        self.relu = nn.ReLU()
        
    def get_gradients(self, q,p):
        state = torch.cat([q,p], dim = 1)
        state.requires_grad_(True)
        H = self.out(self.relu(self.h_dense2(self.relu(self.h_dense1(state)))))
        dH = torch.autograd.grad(
            outputs = H.sum(),
            inputs = state,
            create_graph = True
        )[0]
        dq = dH[:, self.feature_dim:]        # ∂H/∂p
        dp = -dH[:, :self.feature_dim]       # -∂H/∂q

        return dq, dp
    
    def forward(self, x):
    # x: (B, T, D)
        q = x[:, :, -1]             # (B, D)
        p = x[:, :, -1] - x[:, :, -2]   
        
        q = q.detach().requires_grad_(True)
        p = p.detach().requires_grad_(True)# (B, D)

        for _ in range(self.steps):
            dq, dp = self.get_gradients(q, p)

            p = p + 0.5 * self.dt * dp
            q = q + self.dt * dq

            _, dp_final = self.get_gradients(q, p)
            p = p + 0.5 * self.dt * dp_final

        return torch.cat([q, p], dim=-1)
    
#output : Batch, [q,p](2*feature_dim)


#input : Batch, hnn_dim
class HNNdecoder(nn.Module):
    
    def __init__(self,feat_phy,hnn_dim, output_steps):
        super().__init__()
        
        self.feat_phy = feat_phy
        self.output_steps = output_steps
        self.hnn_dim = hnn_dim
        
        self.dense = nn.Linear(hnn_dim, hnn_dim*output_steps)
        self.conv = nn.Conv1d(hnn_dim, feat_phy, kernel_size = 1)
        self.linear_activation = nn.Identity()
        
    def forward(self, phy_features):
        x = phy_features
        B = x.shape[0]
        x = self.dense(x)
        x = torch.tanh(x)
        z = x.view(B, self.hnn_dim, self.output_steps)
        if(self.feat_phy > 0):

            out = self.conv(z)
            out = out
            out = self.linear_activation(out)
        else:
            out = torch.zeros((B, 0, self.output_steps), device = x.device)
        return out

#output : Batch, feature_reconstructed(feat_phy), time_steps_reconstructed
