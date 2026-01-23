import tensorflow as tf
from tqdm import tqdm
import numpy as np
from .losses import encoder_loss, discriminator_loss
from .masking import mix_features

# Preferred persona namespace
namespace = tf.keras

class DualAnchorACAETrainer:
    def __init__(self, encoder, decoder, discriminator, config, topology):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.topo = topology 

        # Hyperparameters from Config
        self.latent_dim = config["latent_dim"]
        self.lr = config["lr"]
        
        # Loss Weights from Config
        self.lambda_d = config["lambda_d"]
        self.lambda_e = config["lambda_e"]
        self.recon_weight = config["recon_weight"]
        self.alpha_vpo = config["alpha_vpo"]
        self.sentinel_weight = config["sentinel_weight"]
        
        # Training Control
        self.patience = config["patience"]
        self.batch_size = config["batch_size"]

        # Optimizers
        self.enc_dec_optimizer = namespace.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0)
        self.disc_optimizer = namespace.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0)

        # Metrics
        self.recon_metric = namespace.metrics.Mean(name="recon_mean")
        self.disc_metric = namespace.metrics.Mean(name="disc_mean")
        self.enc_metric = namespace.metrics.Mean(name="enc_mean")

    @tf.function()
    def _train_step(self, phy_packed, res_windows):
        # 1. Unpack Physics Views
        anchor_phy = phy_packed[:, 0, :, :]        
        aug_views_phy = phy_packed[:, 1:5, :, :] 
        jerk_phy = phy_packed[:, 5, :, :] 
        
        with tf.GradientTape(persistent=True) as tape:
            # 2. Forward Pass
            z_sys, z_res, z_comb = self.encoder([anchor_phy, res_windows], training=True)
            recons_phy, recons_res = self.decoder([z_sys, z_res], training=True)
            
            # 3. Physics Loss (Hamiltonian Branch)
            j_abs = tf.abs(jerk_phy)
            j_thresh = tf.reduce_mean(j_abs) + tf.math.reduce_std(j_abs)
            vpo_weights = tf.where(j_abs > j_thresh, self.alpha_vpo, 1.0)
            if self.topo.idx_phy.shape[0] > 0:
                recon_phy_loss = tf.reduce_mean(tf.square(anchor_phy - recons_phy) * vpo_weights)
            else:
                recon_phy_loss= tf.constant(0.0, dtype=tf.float32)

            # 4. Residual & Sentinel Loss
            recon_res_loss = tf.reduce_mean(tf.square(res_windows - recons_res))
            
            # Check if dead sensors exist to avoid dtype errors and NaN loss
            if len(self.topo.res_to_dead_local) > 0:
                # Explicit cast to int32 prevents the float32 DataType mismatch
                dead_idx = tf.cast(self.topo.res_to_dead_local, tf.int32)
                dead_recons = tf.gather(recons_res, dead_idx, axis=-1)
                sentinel_loss = tf.reduce_mean(tf.square(dead_recons - 0.0))
            else:
                # Provide a zero scalar if no dead sensors are present
                sentinel_loss = tf.constant(0.0)
            
            total_recon_loss = recon_phy_loss + recon_res_loss + (self.sentinel_weight * sentinel_loss)

            # 5. Joint-Manifold Adversarial Task
            z_pos_all, alpha_all = [], []
            for i in range(4):
                view_phy = aug_views_phy[:, i, :, :]
                z_s_aug, _, _ = self.encoder([view_phy, res_windows], training=True)
                z_mixed, alpha = mix_features(z_sys, z_s_aug)
                z_mixed_comb = tf.concat([z_mixed, z_res], axis=-1)
                z_pair = tf.concat([z_comb, z_mixed_comb], axis=1) 
                
                z_pos_all.append(z_pair)
                alpha_all.append(alpha)

            z_mixed_pos = tf.concat(z_pos_all, axis=0)
            alpha_pos = tf.concat(alpha_all, axis=0)
            
            z_neg_mixed, beta_neg = mix_features(z_comb, tf.random.shuffle(z_comb))
            z_neg_pair = tf.concat([z_comb, z_neg_mixed], axis=1) 

            d_out_pos = self.discriminator(z_mixed_pos, training=True)
            d_out_neg = self.discriminator(z_neg_pair, training=True)

            loss_disc = discriminator_loss(d_out_pos, d_out_neg, alpha_pos, beta_neg)
            loss_enc = encoder_loss(d_out_pos, d_out_neg, beta_neg)
            
            total_loss = (self.recon_weight * total_recon_loss + 
                          self.lambda_d * loss_disc + 
                          self.lambda_e * loss_enc)

        # Optimization
        enc_dec_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        self.enc_dec_optimizer.apply_gradients(zip(tape.gradient(total_loss, enc_dec_vars), enc_dec_vars))
        self.disc_optimizer.apply_gradients(zip(tape.gradient(loss_disc, self.discriminator.trainable_variables), 
                                                self.discriminator.trainable_variables))
        
        return total_recon_loss, loss_disc, loss_enc

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