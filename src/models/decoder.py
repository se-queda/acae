import torch
import torch.nn as nn
from .fno import FNOdecoder
from .hnn import HNNdecoder


#input: Batch, [res_latent: (B, fno_dim), phy_latent: (B, hnn_dim)]
class DualDecoder(nn.Module):
    def __init__(self, hnn_dim, fno_dim, feat_phy, feat_res, f_modes, fno_blocks, output_steps):
        super().__init__()
        self.feat_phy = feat_phy
        self.hnn_dim = hnn_dim
        
        self.FNOdecoder = FNOdecoder(feat_res, fno_dim, f_modes, fno_blocks, output_steps = output_steps)
        self.HNNdecoder = HNNdecoder(feat_phy, hnn_dim, output_steps = output_steps)
        
        
        
        
    def forward(self, phy_latent, res_latent):
        x = phy_latent
        y = res_latent
        
        
        restored_res = self.FNOdecoder(y)
        
        restored_phy = self.HNNdecoder(x)
        
        return restored_phy, restored_res

       

#output: Batch, [restored_phy(Batch, window_size, feat_phy), restored_res(Batch, window_size, feat_res)]