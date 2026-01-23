import torch
import torch.nn as nn
import torch.nn.functional as F
from fno import FNOencoder
from hnn import HNN

class TCNResidualBlock(nn.Module):
    def __init__(self, feat_phy, hnn_dim, kernel_size, dilation, dropout=0.1):
        super().__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            feat_phy, hnn_dim,
            kernel_size,
            padding=self.pad,
            dilation=dilation
        )

        self.conv2 = nn.Conv1d(
            hnn_dim, hnn_dim,
            kernel_size,
            padding=self.pad,
            dilation=dilation
        )

        self.skip_conv = (
            nn.Conv1d(feat_phy, hnn_dim, 1)
            if feat_phy != hnn_dim else nn.Identity()
        )

        self.layer_norm = nn.LayerNorm(hnn_dim)
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x):
        # Residual path
        residual = self.skip_conv(x)

        # ---- Conv block 1 (causal) ----
        out = self.conv1(x)
        out = out[:, :, :-self.pad]        # 🔥 causal trim
        out = F.relu(out)
        out = self.dropout(out)

        # ---- Conv block 2 (causal) ----
        out = self.conv2(out)
        out = out[:, :, :-self.pad]        # 🔥 causal trim
        out = self.dropout(out)

        # ---- Residual + normalization ----
        out = out + residual
        out = self.layer_norm(out.transpose(1, 2)).transpose(1, 2)

        return out


# Input: (Batch, feature, timestep)

class TCN(nn.Module):
    def __init__(self, feat_phy, hnn_dim, kernel_size=2, num_layers=6, dropout=0.1):
        super().__init__()

        self.first_conv = nn.Conv1d(feat_phy, hnn_dim, 1)

        dilations = [2 ** i for i in range(num_layers)]  # [1,2,4,8,16,32]
        self.blocks = nn.ModuleList([
            TCNResidualBlock(hnn_dim, hnn_dim, kernel_size, d, dropout)
            for d in dilations
        ])

    def forward(self, x):
        # x: (B, C, T)
        x = self.first_conv(x)

        for block in self.blocks:
            x = block(x)

        return x  # (B, hnn_dim, T)

# Output: (Batch, hnn_dim, Timestep)



#input: Batch, Feature, Timestamp
class DualEncoder(nn.Module):
    def __init__(self, hnn_dim, fno_dim, feat_phy, feat_res, f_modes, fno_blocks, integration_steps, dt):
        super().__init__()
        self.feat_phy = feat_phy
        self.hnn_dim = hnn_dim
        self.FNOencoder = FNOencoder(feat_res, fno_dim, f_modes, fno_blocks)
        self.HNNlayer = HNN(hnn_dim, integration_steps, dt)
        self.HNNProj = TCN(feat_phy, hnn_dim)
        
        self.dense1 = nn.Linear(hnn_dim, hnn_dim)
        self.dense2 = nn.Linear(hnn_dim, hnn_dim)
        self.drop = nn.Dropout(0.01)
        
        
        
    def forward(self, phy_features, res_features):
        x = phy_features
        y = res_features
        B = x.shape[0]
        
        if(self.feat_phy > 0 ): 
            x = self.HNNProj(x)
            flow = self.HNNlayer(x)
            z_phy  = self.dense1(flow)
            z_phy = torch.tanh(z_phy)
            z_phy = self.drop(z_phy)
            z_phy = self.dense2(z_phy)
            z_phy = z_phy.mean(dim=2) 
        else:
            z_phy = torch.zeros(
                B, self.hnn_dim,
                device=x.device
            )
        
        z_res = self.FNOencoder(y)
        z_combined = torch.cat([z_phy, z_res], dim = -1)
        return z_combined 

#output: Bathc, Z_Combined(hnn_dim + fno_dim == L)