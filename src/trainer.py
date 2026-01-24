
from models import DualEncoder as encoder
from models import DualDecoder as decoder
from models import Discriminator as discriminator
from fno_config import fno_config
from hnn_config import hnn_config
from global_config import global_config

from masking import mix_features
from losses import discriminator_loss, encoder_loss

import torch 
import torch.nn as nn
import torch.nn.functional as F


import tqdm
import numpy as np

class DualAnchorACAETrainer:
    def __init__(self, encoder, decoder, discriminator, config, topology):

        self.topo = topology 
        
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
        self.patience = global_config.get('patience', 20)
        self.recon_weight = global_config.get('recon_weight', 1.0)
        self.lambda_d = global_config.get('lambda_d', 1.0)
        self.lambda_e = global_config.get('lambda_e', 1.0)
        self.sentinel_weight = global_config.get('sentinel_weight', 10.0)
        self.alpha_vpo = global_config.get('alpha_vpo', 150.0)
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
        jerk_thres = torch.mean(j_abs) + torch.std(j_abs)
        vpo_weights = torch.where(j_abs>jerk_thres, torch.full_like(j_abs, self.alpha_vpo), torch.ones_like(j_abs))
        
        
        if(self.feat_phy>0):
            recon_phy_loss = torch.mean(F.mse_loss(anchor_phy, recons_phy, reduction='none') * vpo_weights)
        else:
            recon_phy_loss = torch.tensor(0.0, device=self.device)

        recon_res_loss = torch.mean(F.mse_loss(res_windows, recons_res, reduction='none'))
        
        if(len(self.topo.res_to_dead_local) > 0):
            dead_idx = torch.tensor(self.topo.read_dead_local, dtype=torch.long, device=self.device)
            dead_recons = recons_res[:, dead_idx, :]
            sentinel_loss = sentinel_loss = F.mse_loss(dead_recons, torch.zeros_like(dead_recons))

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
        return {
            "total_loss": total_loss.detach(),
            "recon_phy": recon_phy_loss.detach(),
            "recon_res": recon_res_loss.detach(),
            "sentinel": sentinel_loss.detach(),
            "loss_disc": loss_disc.detach(),
            "loss_enc": loss_enc.detach(),
        }


    def fit(self, train_ds, val_ds=None, epochs=200):
        best_val_loss = float("inf")
        wait = 0 
        
        for epoch in range(epochs):
            self.recon_metric.reset_state()
            self.disc_metric.reset_state()
            self.enc_metric.reset_state()
            
            bar = tqdm(train_ds, desc=f"Epoch {epoch+1}", unit="batch")
            for phy_batch, res_batch in bar:
                r_loss, d_loss, e_loss = self._train_step(phy_batch, res_batch)
                self.recon_metric.update_state(r_loss)
                self.disc_metric.update_state(d_loss)
                self.enc_metric.update_state(e_loss)
                
                bar.set_postfix({"Recon": f"{r_loss:.4f}", "Disc": f"{d_loss:.4f}"})

            if val_ds is not None:
                val_losses = []
                for v_phy, v_res in val_ds:
                    # 1. Forward Pass (No training)
                    v_phy_anchor = v_phy[:, 0, :, :] 
                    z_s, z_r, _ = self.encoder([v_phy_anchor, v_res], training=False)
                    h_phy, h_res = self.decoder([z_s, z_r], training=False)
                    
                    # 2. FIXED: Calculate MSE for each branch separately to avoid shape broadcast errors
                    mse_phy = tf.reduce_mean(tf.square(v_phy_anchor - h_phy))
                    mse_res = tf.reduce_mean(tf.square(v_res - h_res))
                    
                    # 3. Add the scalars together
                    val_losses.append((mse_phy + mse_res).numpy())
                
                current_val_loss = float(np.mean(val_losses))
                print(f"✅ Val MSE: {current_val_loss:.4f} | Best: {best_val_loss:.4f}")
                
                if current_val_loss < best_val_loss:
                    best_val_loss, wait = current_val_loss, 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        print(f"🛑 Early stopping at epoch {epoch+1}")
                        break
                    
    def reconstruct(self, test_final, batch_size=128):
        print("🔍 Generating Point-wise Reconstructions...")
        phy_views = tf.cast(test_final['phy'], tf.float32) 
        res_data  = tf.cast(test_final['res'], tf.float32)
        
        # Determine anchor
        phy_anchor = phy_views[:, 0, :, :] if len(phy_views.shape) == 4 else phy_views

        # Batched inference to prevent GPU OOM
        z_sys_list, z_res_list = [], []
        for i in range(0, len(phy_anchor), batch_size):
            p_batch = phy_anchor[i:i+batch_size]
            r_batch = res_data[i:i+batch_size]
            zs, zr, _ = self.encoder([p_batch, r_batch], training=False)
            z_sys_list.append(zs)
            z_res_list.append(zr)
        
        z_sys = tf.concat(z_sys_list, axis=0)
        z_res = tf.concat(z_res_list, axis=0)
        
        # Batched decoding
        phy_hat_list, res_hat_list = [], []
        for i in range(0, len(z_sys), batch_size):
            ph, rh = self.decoder([z_sys[i:i+batch_size], z_res[i:i+batch_size]], training=False)
            phy_hat_list.append(ph)
            res_hat_list.append(rh)
            
        return {
            "phy_orig": phy_anchor.numpy(),
            "phy_hat": tf.concat(phy_hat_list, axis=0).numpy(),
            "res_orig": res_data.numpy(),
            "res_hat": tf.concat(res_hat_list, axis=0).numpy()
        }