import torch
import torch.nn as nn
from modulus.models.fno import fno
from modulus.models.layers import get_activation

#input: Batch, features(feat_res), time steps

class FNOencoder(nn.Module):
    
    def __init__(self, feat_res, fno_dim, f_modes, fno_blocks):
        super().__init__()

        
        self.encoder = fno.FNO1DEncoder(
            in_channels=feat_res, #no of features
            num_fno_layers=fno_blocks,
            fno_layer_size=fno_dim, 
            num_fno_modes=f_modes,
            padding=8,
            padding_type="constant",
            coord_features=False,
            activation_fn=get_activation("gelu"),
        )

        # Projection head (YOU own this)
        self.proj = nn.Linear(fno_dim, fno_dim)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.gap(x).squeeze(-1)
        z_res = self.proj(x)
        
        return z_res
#output: Batch, fno_dim  


#input : Batch, fno_dim
class FNOdecoder(nn.Module):
    
    def __init__(self, feat_res, fno_dim, f_modes, fno_blocks, output_steps):
        super().__init__()
        
        self.fno_dim = fno_dim
        self.output_steps = output_steps
        
        self.proj = nn.Linear(fno_dim, fno_dim * output_steps)
        # Latent-space FNO propagation
        self.fno = fno.FNO1DDecoder(
            in_channels=fno_dim,
            out_channels=fno_dim,
            num_fno_layers=fno_blocks,
            fno_layer_size=fno_dim,
            num_fno_modes=f_modes,
            padding=8,
            padding_type="constant",
            activation_fn=get_activation("gelu"),
        )
        
        self.out = nn.Conv1d(fno_dim, feat_res, kernel_size = 1)
        
    def forward(self, x):
        
        B =x.shape[0]
        x = self.proj(x)
        z = x.view(B, self.fno_dim, self.output_steps)
        z = self.fno(z)
        out = self.out(z)
        
        return out.permute(0,2,1)


#output : Batch, feature_reconstructed(feat_res), time_steps_reconstructed