import tensorflow as tf
from tqdm import tqdm
import numpy as np
from .losses import encoder_loss, discriminator_loss
from .masking import mix_features

namespace = tf.keras

class DualAnchorACAETrainer:
    def __init__(self, encoder, decoder, discriminator, config, topology):
        """
        encoder: DualHeadEncoder outputting [z_sys, z_res, z_combined]
        decoder: DualHeadDecoder outputting [out_sys, out_res]
        """
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.topo = topology # MachineTopology object from router.py

        # Hyperparameters
        self.latent_dim = config.get("latent_dim", 256) 
        self.lr = config.get("lr", 1e-4)
        
        # Loss Weights
        self.lambda_d = config.get("lambda_d", 1.0)
        self.lambda_e = config.get("lambda_e", 1.0)
        self.recon_weight = config.get("recon_weight", 1.0)
        self.alpha_vpo = config.get("alpha_vpo", 10.0) # Physics weighting
        self.sentinel_weight = config.get("sentinel_weight", 5.0) # Dead feature weighting

        # Optimizers
        self.enc_dec_optimizer = namespace.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0)
        self.disc_optimizer = namespace.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0)

        # Metrics
        self.recon_metric = namespace.metrics.Mean(name="recon_mean")
        self.disc_metric = namespace.metrics.Mean(name="disc_mean")
        self.enc_metric = namespace.metrics.Mean(name="enc_mean")

    @tf.function
    def _train_step(self, phy_packed, res_windows):
        """
        phy_packed: (N, 6, 64, F_phy) - Views for Hamiltonian Branch
        res_windows: (N, 64, F_res)  - Windows for Residual Branch
        """
        # Unpack Physics Views
        anchor_phy = phy_packed[:, 0, :, :]        
        aug_views_phy = phy_packed[:, 1:5, :, :] 
        jerk_phy = phy_packed[:, 5, :, :] 
        
        with tf.GradientTape(persistent=True) as tape:
            # 1. FORWARD PASS: Dual-Anchor Inference
            z_sys, z_res, z_comb = self.encoder([anchor_phy, res_windows], training=True)
            recons_phy, recons_res = self.decoder([z_sys, z_res], training=True)
            
            # 2. PHYSICS LOSS (Hamiltonian Branch)
            j_abs = tf.abs(jerk_phy)
            j_thresh = tf.reduce_mean(j_abs) + tf.math.reduce_std(j_abs)
            vpo_weights = tf.where(j_abs > j_thresh, self.alpha_vpo, 1.0)
            recon_phy_loss = tf.reduce_mean(tf.square(anchor_phy - recons_phy) * vpo_weights)
            
            # 3. RESIDUAL & SENTINEL LOSS (Identity Branch)
            recon_res_loss = tf.reduce_mean(tf.square(res_windows - recons_res))
            
            # THE SENTINEL BAKE: Force dead features (via topo local indices) to absolute zero
            dead_recons = tf.gather(recons_res, self.topo.res_to_dead_local, axis=-1)
            sentinel_loss = tf.reduce_mean(tf.square(dead_recons - 0.0))
            
            total_recon_loss = recon_phy_loss + recon_res_loss + (self.sentinel_weight * sentinel_loss)

            # 4. UPGRADED JOINT-MANIFOLD ADVERSARIAL TASK
            z_pos_all, alpha_all = [], []
            for i in range(4):
                view_phy = aug_views_phy[:, i, :, :]
                # Challenge the physics head while keeping the residual head stable
                z_s_aug, _, _ = self.encoder([view_phy, res_windows], training=True)
                
                # MIXING UPGRADE: Interpolate ONLY the physics latent
                # This ensures the HNN learns to hallucinate physics within a stable machine context
                z_mixed, alpha = mix_features(z_sys, z_s_aug)
                
                # Reconstruct the mixed Joint Latent [Mixed_Sys, Original_Res]
                z_mixed_comb = tf.concat([z_mixed, z_res], axis=-1)
                
                z_pos_all.append(tf.concat([z_comb, z_mixed_comb], axis=1))
                alpha_all.append(alpha)

            z_mixed_pos = tf.concat(z_pos_all, axis=0)
            alpha_pos = tf.concat(alpha_all, axis=0)
            
            # Negative Pair: Mix anchor with a shuffled version of itself
            z_neg_mixed, beta_neg = mix_features(z_comb, tf.random.shuffle(z_comb))
            z_neg_pair = tf.concat([z_comb, z_neg_mixed], axis=1)

            # Discriminator judging the "Purity" of the joint state
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
            """
            Accepts pre-built datasets from build_tf_datasets.
            Each batch is a tuple: (phy_batch, res_batch)
            """
            best_val_loss = float("inf")
            patience = 15  # Increased slightly for stability
            wait = 0 
            
            for epoch in range(epochs):
                self.recon_metric.reset_state()
                self.disc_metric.reset_state()
                self.enc_metric.reset_state()
                
                # Use tqdm bar over the dataset
                bar = tqdm(train_ds, desc=f"Epoch {epoch+1}", unit="batch")
                for phy_batch, res_batch in bar:
                    r_loss, d_loss, e_loss = self._train_step(phy_batch, res_batch)
                    
                    self.recon_metric.update_state(r_loss)
                    self.disc_metric.update_state(d_loss)
                    self.enc_metric.update_state(e_loss)
                    
                    bar.set_postfix({
                        "Recon": f"{r_loss:.4f}", 
                        "Disc": f"{d_loss:.4f}",
                        "Enc": f"{e_loss:.4f}"
                    })

                # Validation logic using the zipped val_ds
                if val_ds is not None:
                    val_losses = []
                    for v_phy, v_res in val_ds:
                        # Logic to calculate val_recon across both branches
                        # We only care about the first view (index 0) of v_phy for validation
                        v_phy_anchor = v_phy[:, 0, :, :] 
                        z_s, z_r, _ = self.encoder([v_phy_anchor, v_res], training=False)
                        h_phy, h_res = self.decoder([z_s, z_r], training=False)
                        
# FIX: Calculate MSE for each branch separately, THEN add the scalars
                    mse_phy = tf.reduce_mean(tf.square(v_phy_anchor - h_phy))
                    mse_res = tf.reduce_mean(tf.square(v_res - h_res))
                    
                    # Now they are both scalars, adding them is safe
                    val_losses.append((mse_phy + mse_res).numpy())
                    
                    current_val_loss = float(np.mean(val_losses))
                    print(f"✅ Val Total Recon (MSE): {current_val_loss:.4f}")
                    
                    if current_val_loss < best_val_loss:
                        best_val_loss, wait = current_val_loss, 0
                    else:
                        wait += 1
                        if wait >= patience:
                            print(f"🛑 Early stopping at epoch {epoch+1}")
                            break
                    
    def reconstruct(self, test_final):
            # Unpack from test_final (standardizing on 'phy' and 'res' keys)
            phy_views = tf.cast(test_final['phy'], tf.float32) 
            res_data  = tf.cast(test_final['res'], tf.float32)
            
            # Slicing: If test_phy is 4D (N, 6, 64, F), take view 0.
            # If it's already 3D (N, 64, F), it stays as is.
            if len(phy_views.shape) == 4:
                phy_anchor = phy_views[:, 0, :, :]
            else:
                phy_anchor = phy_views

            z_sys, z_res, _ = self.encoder([phy_anchor, res_data], training=False)
            recons_phy, recons_res = self.decoder([z_sys, z_res], training=False)
            
            return {
                "phy_orig": phy_anchor.numpy(), # 3D only for the scorer
                "phy_hat": recons_phy.numpy(),
                "res_orig": res_data.numpy(),
                "res_hat": recons_res.numpy()
            }