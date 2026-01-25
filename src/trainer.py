
from src.models import DualEncoder as encoder
from src.models import DualDecoder as decoder
from src.models import Discriminator as discriminator
from src.configs.fno_config import fno_config
from src.configs.hnn_config import hnn_config
from src.configs.global_config import global_config

from src.masking import mix_features
from src.losses import discriminator_loss, encoder_loss

import torch 
import torch.nn as nn
import torch.nn.functional as F


from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class DualAnchorACAETrainer:
    def __init__(self, encoder, decoder, discriminator, hnn_config, fno_config, global_config, topology):

        self.topo = topology 
        self.hnn_config = hnn_config
        self.fno_config = fno_config
        self.global_config = global_config
        #sensor numbers
        self.num_phy_sensors = topology.num_phy_sensors
        self.num_res_sensors = topology.num_res_sensors
        self.topo = topology
        #hnn hyperparameters
        self.hnn_dim = hnn_config['hnn_dim']
        self.integration_steps = hnn_config['integration_steps']
        self.dt = hnn_config['dt']
        
        
        #fno hyperparameters
        self.fno_dim = fno_config['fno_dim']
        self.f_modes = fno_config['f_modes']
        self.fno_blocks = fno_config['fno_blocks']
        
        #feature_length
        self.feat_phy = topology.num_phy_features
        self.feat_res = topology.num_res_features
        
        # Training hyperparameters
        self.lr = global_config['learning_rate']
        self.patience = global_config.get('patience')
        self.recon_weight = global_config.get('recon_weight')
        self.lambda_d = global_config.get('lambda_d')
        self.lambda_e = global_config.get('lambda_e')
        self.sentinel_weight = global_config.get('sentinel_weight')
        self.alpha_vpo = global_config.get('alpha_vpo')
        self.window_size = global_config['window_size']

        # Model Initialization
        self.encoder = encoder(hnn_dim=self.hnn_dim,
                               fno_dim=self.fno_dim,
                               feat_phy=self.feat_phy,
                               feat_res=self.feat_res,
                               f_modes=self.f_modes,
                               fno_blocks=self.fno_blocks,
                               integration_steps=self.integration_steps,
                               dt=self.dt)
        self.decoder = decoder(hnn_dim=self.hnn_dim,
                               fno_dim=self.fno_dim,
                               feat_phy=self.feat_phy,
                               feat_res=self.feat_res,
                               f_modes=self.f_modes,
                               fno_blocks=self.fno_blocks,
                               output_steps=self.window_size)

        self.discriminator = discriminator(input_dim=2 * self.hnn_dim)

        
        # Optimizers
        self.enc_dec_optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        
        #device settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.discriminator.to(self.device)  
        self.grad_clip = 1.0
        
        #debub
        self.debug = True
        if self.debug:
            print("✅ Using Device:", self.device)
            
            
        #tensorboard
        self.writer = SummaryWriter(log_dir="runs/dual_anchor_acae")

    
    def train_step(self, phy_packed, res_windows):
    # ===============================
    # 0. Move data to device
    # ===============================
        phy_packed = phy_packed.to(self.device)
        res_windows = res_windows.to(self.device)
    # ===============================
    # 1. Unpack physics views
    # ===============================
        anchor_phy = phy_packed[:,0]
        aug_views_phy = phy_packed[:,1:5]
        jerk_phy = phy_packed[:,5]          
    # ===============================
    # 2. Forward pass (encoder + decoder)
    # ===============================
        z_comb = self.encoder(anchor_phy, res_windows)
        z_sys = z_comb[:, :self.hnn_dim]
        z_res = z_comb[:, self.hnn_dim:]
        recons_phy, recons_res = self.decoder(z_sys, z_res)
        
        #debu mode
        if self.debug:
            assert z_sys.shape[1] == self.hnn_dim
            assert z_res.shape[1] == self.fno_dim
            assert recons_phy.shape == anchor_phy.shape
            assert recons_res.shape == res_windows.shape
    # ===============================
    # 3. Reconstruction losses
    #   - physics (jerk-weighted)
    #   - residual
    #   - sentinel
    # ===============================
        j_abs = torch.abs(jerk_phy)
        jerk_thres = torch.mean(j_abs) + torch.std(j_abs) + 1e-6 
        vpo_weights = torch.where(j_abs>jerk_thres, torch.full_like(j_abs, self.alpha_vpo), torch.ones_like(j_abs))
        
        
        if(self.feat_phy>0):
            recon_phy_loss = torch.mean(F.mse_loss(anchor_phy, recons_phy, reduction='none') * vpo_weights)
        else:
            recon_phy_loss = torch.tensor(0.0, device=self.device)

        recon_res_loss = torch.mean(F.mse_loss(res_windows, recons_res, reduction='none'))
        
        if(len(self.topo.res_to_dead_local) > 0):
            dead_idx = torch.tensor(self.topo.res_to_dead_local, dtype=torch.long, device=self.device)
            dead_recons = recons_res[:, dead_idx, :]
            sentinel_loss = F.mse_loss(dead_recons, torch.zeros_like(dead_recons))

        else:
            sentinel_loss = torch.tensor(0.0, device=self.device)
        
        total_recon_loss = recon_phy_loss + recon_res_loss + (self.sentinel_weight * sentinel_loss)
        

    # ===============================
    # 4. Adversarial manifold construction
    #   - multi-view positive pairs
    #   - negative pairs
    # ===============================

        z_pos_all, alpha_all = [], []
        B = z_sys.size(0)

        # -----------------------------
        # Positive composites (4 views)
        # -----------------------------
        for i in range(4):
            view_phy = aug_views_phy[:, i]  # (B, C_phy, T) [BCT]

            # encode augmented view; we only care about physics latent
            z_comb_aug = self.encoder(view_phy, res_windows)     # (B, L)
            z_sys_aug  = z_comb_aug[:, :self.hnn_dim]            # (B, hnn_dim)

            # mix physics latents (ACAE)
            z_mixed_sys, alpha = mix_features(z_sys, z_sys_aug)  # (B, hnn_dim), (B, 1)

            # discriminator sees PAIR: [anchor_phy , mixed_phy]
            z_pair = torch.cat([z_sys, z_mixed_sys], dim=1)      # (B, 2*hnn_dim)

            z_pos_all.append(z_pair)
            alpha_all.append(alpha)

        z_mixed_pos = torch.cat(z_pos_all, dim=0)   # (4B, 2*hnn_dim)
        alpha_pos   = torch.cat(alpha_all, dim=0)   # (4B, 1)

        # -----------------------------
        # Negative composites (ACAE)
        # -----------------------------
        perm = torch.randperm(B, device=z_sys.device)
        z_sys_shuf = z_sys[perm]                                 # (B, hnn_dim)

        z_neg_mixed, beta_neg = mix_features(z_sys, z_sys_shuf)  # (B, hnn_dim), (B, 1)

        z_neg_pair = torch.cat([z_sys, z_neg_mixed], dim=1)      # (B, 2*hnn_dim)

    # ===============================
    # 5. Discriminator loss
    # ===============================
        self.disc_optimizer.zero_grad()
        d_out_pos = self.discriminator(z_mixed_pos.detach())  # (4B, 2)
        d_out_neg = self.discriminator(z_neg_pair.detach()) 
        # (B, 2
        loss_disc = discriminator_loss(d_out_pos, d_out_neg, alpha_pos, beta_neg)
        loss_disc.backward()
        self.disc_optimizer.step()
    # ===============================
    # 6. Encoder adversarial loss
    # ===============================
        self.enc_dec_optimizer.zero_grad()
        d_out_pos = self.discriminator(z_mixed_pos)  # (4B, 2)
        d_out_neg = self.discriminator(z_neg_pair)  # (B, 2
        loss_enc = encoder_loss(d_out_pos, d_out_neg, beta_neg)
        
        total_loss = self.recon_weight * total_recon_loss + self.lambda_e * loss_enc
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), self.grad_clip)
        self.enc_dec_optimizer.step()
    # ===============================
    # 8. Return scalars
    # ===============================
        if self.debug:
            assert not torch.isnan(total_loss)
            assert not torch.isnan(loss_disc)

        return {
            "total_loss": total_loss.detach(),
            "recon_phy": recon_phy_loss.detach(),
            "recon_res": recon_res_loss.detach(),
            "sentinel": sentinel_loss.detach(),
            "loss_disc": loss_disc.detach(),
            "loss_enc": loss_enc.detach(),
        }
    
    
    def save_checkpoint(self, epoch, best_val_loss, path="checkpoint.pt"):
        torch.save({
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "enc_dec_optimizer": self.enc_dec_optimizer.state_dict(),
            "disc_optimizer": self.disc_optimizer.state_dict(),
        }, path)

    
    
    def fit(self, train_ds, val_ds=None, epochs=200):
        best_val_loss = float("inf")
        wait = 0
        
        
        for epoch in range(epochs):
            
            #Train
            self.encoder.train()
            self.decoder.train()
            self.discriminator.train()
            
            
            epoch_recon = 0.0
            epoch_disc = 0.0
            epoch_enc = 0.0
            num_batches = 0
            
            bar = tqdm(train_ds, desc = f"{epoch+1}", unit = 'batch')
            for phy_batch, res_batch in bar:
                
                out = self.train_step(phy_batch, res_batch)
                recon_loss = out["recon_phy"] + out["recon_res"] + out["sentinel"]
                disc_loss  = out["loss_disc"]
                enc_loss   = out["loss_enc"]
                
                epoch_recon += recon_loss.item()
                epoch_disc  += disc_loss.item()
                epoch_enc   += enc_loss.item()
                num_batches += 1
                
                bar.set_postfix(
                    {
                        "Recon":f"{epoch_recon/num_batches:.4f}",
                        "Disc": f"{epoch_disc/num_batches:.4f}",
                        "enc": f"{epoch_enc/num_batches:.4f}"
                    }
                )
            self.writer.add_scalar("train/recon_loss", epoch_recon / num_batches, epoch)
            self.writer.add_scalar("train/disc_loss",  epoch_disc  / num_batches, epoch)
            self.writer.add_scalar("train/enc_loss",   epoch_enc   / num_batches, epoch)

                    
            #validation
            if val_ds is not None:
                self.encoder.eval()
                self.decoder.eval()

                val_losses = []

                with torch.no_grad():
                    for v_phy, v_res in val_ds:
                        v_phy = v_phy.to(self.device)
                        v_res = v_res.to(self.device)

                        anchor = v_phy[:, 0]  # (B, C, T)

                        z_comb = self.encoder(anchor, v_res)
                        z_sys  = z_comb[:, :self.hnn_dim]
                        z_res  = z_comb[:, self.hnn_dim:]

                        h_phy, h_res = self.decoder(z_sys, z_res)

                        mse_phy = F.mse_loss(anchor, h_phy)
                        mse_res = F.mse_loss(v_res, h_res)

                        val_losses.append((mse_phy + mse_res).item())

                current_val_loss = float(np.mean(val_losses))
                tqdm.write(
                    f"Val MSE: {current_val_loss:.4f} | Best: {best_val_loss:.4f}"
                )
                
                self.writer.add_scalar("val/mse", current_val_loss, epoch)

                if(current_val_loss<best_val_loss):
                    best_val_loss = current_val_loss
                    wait = 0
                    self.save_checkpoint(epoch, best_val_loss)
                    tqdm.write("best models saved")
                else:
                    wait +=1
                    if(wait>=self.patience):
                        tqdm.write(f"early stopping {epoch+1}")
                        break

        
        
        
    def reconstruct(self, test_final, batch_size=128):
        print("🔍 Generating Point-wise Reconstructions...")

        phy_views = test_final["phy"].float().to(self.device)
        res_data  = test_final["res"].float().to(self.device)

        # Determine anchor view
        if phy_views.dim() == 4:   # (B, V, C, T)
            phy_anchor = phy_views[:, 0]   # (B, C, T)
        else:
            phy_anchor = phy_views         # already (B, C, T)
        self.encoder.eval()
        self.decoder.eval()
        
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        # Batched encoding
        z_sys_list, z_res_list = [], []
        with torch.enable_grad():  # CHANGED: HNN uses autograd.grad, so encoding needs grads enabled.
            for i in range(0, phy_anchor.size(0), batch_size):
                p_batch = phy_anchor[i:i + batch_size]
                r_batch = res_data[i:i + batch_size]
                z_comb = self.encoder(p_batch, r_batch)
                z_sys  = z_comb[:, :self.hnn_dim]
                z_res  = z_comb[:, self.hnn_dim:]
                z_sys_list.append(z_sys.detach())  # CHANGED: detach to keep recon deterministic and avoid graph growth.
                z_res_list.append(z_res.detach())  # CHANGED: detach to keep recon deterministic and avoid graph growth.
            z_sys = torch.cat(z_sys_list, dim=0)
            z_res = torch.cat(z_res_list, dim=0)

        # Batched decoding
        phy_hat_list, res_hat_list = [], []
        with torch.no_grad():  # CHANGED: decoding does not need gradients.
            for i in range(0, z_sys.size(0), batch_size):
                ph, rh = self.decoder(
                    z_sys[i:i + batch_size],
                    z_res[i:i + batch_size]
                )
                phy_hat_list.append(ph)
                res_hat_list.append(rh)
            phy_hat = torch.cat(phy_hat_list, dim=0)
            res_hat = torch.cat(res_hat_list, dim=0)


        return {
            "phy_orig": phy_anchor.cpu().numpy(),
            "phy_hat":  phy_hat.cpu().numpy(),
            "res_orig": res_data.cpu().numpy(),
            "res_hat":  res_hat.cpu().numpy(),
        }


    
    
    #Pipeline checker functions, not used while training with real data, only used during model debug.


    def _make_dummy_batch(self, B=8):
        """
        Creates a single dummy batch matching trainer expectations.
        """

        T = self.window_size
        C_phy = self.topo.num_phy_features
        C_res = self.topo.num_res_features
        V = 6  # anchor + 4 aug + jerk

        # Physics views: (B, V, C_phy, T)
        phy = torch.randn(B, V, C_phy, T, device=self.device)

        # Residual windows: (B, C_res, T)
        res = torch.randn(B, C_res, T, device=self.device)

        return phy, res

    def debug_sanity_check(self):
        print("🧪 Running full sanity check...")

        phy, res = self._make_dummy_batch(B=4)

        # ---- train_step ----
        out = self.train_step(phy, res)

        print("✅ train_step outputs:")
        for k, v in out.items():
            print(f"  {k}: {float(v):.6f}")

        # ---- reconstruct ----
        dummy_test = {
            "phy": phy[:, :1],   # fake anchor-only input
            "res": res
        }

        recon = self.reconstruct(dummy_test, batch_size=2)

        print("reconstruct outputs:")
        for k, v in recon.items():
            print(f"  {k}: shape = {v.shape}")

        print("successfull")




#dummy topology and dummy checker script.

'''
# DUMMY TOPOLOGY

topology = type("Topology", (), {})()

# global sensor indices
topology.idx_phy  = np.array([0, 1, 2, 3])        # 4 physics sensors
topology.idx_res  = np.array([0, 1, 2, 3, 4, 5])  # 6 residual sensors

# no dead / lone sensors for now
topology.idx_dead = np.array([])
topology.idx_lone = np.array([])

# local mappings (empty → sentinel disabled cleanly)
topology.res_to_dead_local = []
topology.res_to_lone_local = []

# counts (trainer relies on these)
topology.num_phy_features = len(topology.idx_phy)
topology.num_res_features = len(topology.idx_res)

# aliases used elsewhere
topology.num_phy_sensors = topology.num_phy_features
topology.num_res_sensors = topology.num_res_features

trainer = DualAnchorACAETrainer(
    encoder,
    decoder,
    discriminator,
    hnn_config,
    fno_config,
    global_config,
    topology
)

trainer.debug_sanity_check()
'''
