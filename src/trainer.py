import tensorflow as tf
from tqdm import tqdm
from .losses import encoder_loss, discriminator_loss
from .masking import mix_features
import numpy as np

namespace = tf.keras

class ACAETrainer:
    def __init__(self, encoder, decoder, discriminator, config):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

        # Hyperparameters
        self.latent_dim = config.get("latent_dim", 256) 
        self.lr = config.get("lr", 1e-4)
        
        # Loss Weights
        self.lambda_d = config.get("lambda_d", 1.0)
        self.lambda_e = config.get("lambda_e", 1.0)
        self.recon_weight = config.get("recon_weight", 1.0)
        self.alpha_vpo = config.get("alpha_vpo", 10.0)
        self.lambda_mom = config.get("lambda_mom", 0.5)

        # Optimizers
        self.enc_dec_optimizer = namespace.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0)
        self.disc_optimizer = namespace.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0)

        # --- METRICS (FIXED: Added enc_metric) ---
        self.recon_metric = namespace.metrics.Mean(name="recon_mean")
        self.disc_metric = namespace.metrics.Mean(name="disc_mean")
        self.enc_metric = namespace.metrics.Mean(name="enc_mean") # This was missing!

    @tf.function
    def _train_step(self, packed_batch):
        packed_batch = tf.cast(packed_batch, tf.float32)
        anchor = packed_batch[:, 0, :, :]        
        physics_views = packed_batch[:, 1:5, :, :] 
        jerk_w = packed_batch[:, 5, :, :] 
        
        with tf.GradientTape(persistent=True) as tape:
            # Sobolev Reconstruction
            z = self.encoder(anchor, training=True)
            x_hat = self.decoder(z, training=True)
            
            j_abs = tf.abs(jerk_w)
            j_thresh = tf.reduce_mean(j_abs) + tf.math.reduce_std(j_abs)
            vpo_weights = tf.where(j_abs > j_thresh, self.alpha_vpo, 1.0)

            pos_loss = tf.reduce_mean(tf.square(anchor - x_hat) * vpo_weights)
            v_orig = anchor[:, 1:, :] - anchor[:, :-1, :]
            v_hat = x_hat[:, 1:, :] - x_hat[:, :-1, :]
            v_weights = vpo_weights[:, 1:, :]
            mom_loss = tf.reduce_mean(tf.square(v_orig - v_hat) * v_weights)
            
            recon_loss = pos_loss + (self.lambda_mom * mom_loss)

            # Adversarial Contrastive Task
            # Feature combination is the proxy task [cite: 395, 397]
            z_pos_all, alpha_all = [], []
            for i in range(4):
                view = physics_views[:, i, :, :]
                z_aug = self.encoder(view, training=True)
                z_mixed, alpha = mix_features(z, z_aug)
                z_pos_all.append(tf.concat([z, z_mixed], axis=1))
                alpha_all.append(alpha)

            z_mixed_pos = tf.concat(z_pos_all, axis=0)
            alpha_pos = tf.concat(alpha_all, axis=0)
            
            z_neg_mixed, beta_neg = mix_features(z, tf.random.shuffle(z))
            z_neg_pair = tf.concat([z, z_neg_mixed], axis=1)

            d_out_pos = self.discriminator(z_mixed_pos, training=True)
            d_out_neg = self.discriminator(z_neg_pair, training=True)

            loss_disc = discriminator_loss(d_out_pos, d_out_neg, alpha_pos, beta_neg)
            loss_enc = encoder_loss(d_out_pos, d_out_neg, beta_neg)
            
            total_loss = (self.recon_weight * recon_loss + 
                          self.lambda_d * loss_disc + 
                          self.lambda_e * loss_enc)

        enc_dec_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        self.enc_dec_optimizer.apply_gradients(zip(tape.gradient(total_loss, enc_dec_vars), enc_dec_vars))
        self.disc_optimizer.apply_gradients(zip(tape.gradient(loss_disc, self.discriminator.trainable_variables), 
                                                self.discriminator.trainable_variables))
        
        return recon_loss, loss_disc, loss_enc

    def fit(self, train_ds, val_ds=None, epochs=200):
        best_val_loss = float("inf")
        patience = 5 
        wait = 0 
        
        for epoch in range(epochs):
            self.recon_metric.reset_state()
            self.disc_metric.reset_state()
            self.enc_metric.reset_state() # Now safe to call!
            
            bar = tqdm(train_ds, desc=f"Epoch {epoch+1}", unit="batch")
            for batch in bar:
                r_loss, d_loss, e_loss = self._train_step(batch)
                self.recon_metric.update_state(r_loss)
                self.disc_metric.update_state(d_loss)
                self.enc_metric.update_state(e_loss)
                
                bar.set_postfix({
                    "PhysRecon": f"{r_loss:.4f}", 
                    "Disc": f"{d_loss:.4f}",
                    "Enc": f"{e_loss:.4f}"
                })

            if val_ds is not None:
                val_losses = []
                for v_batch in val_ds:
                    v_anchor = tf.cast(v_batch[:, 0, :, :], tf.float32)
                    z_val = self.encoder(v_anchor, training=False)
                    v_hat = self.decoder(z_val, training=False)
                    val_losses.append(tf.reduce_mean(tf.square(v_anchor - v_hat)).numpy())
                val_recon = float(np.mean(val_losses))
                print(f"✅ Val Recon (MSE): {val_recon:.4f}")
                
                if val_recon < best_val_loss:
                    best_val_loss, wait = val_recon, 0
                else:
                    wait += 1
                    if wait >= patience: break
                    
    def reconstruct(self, dataset):
        reconstructions, originals = [], []
        for batch in dataset:
            batch_data = tf.cast(batch, tf.float32)
            z = self.encoder(batch_data, training=False)
            x_hat = self.decoder(z, training=False)
            reconstructions.append(x_hat.numpy())
            originals.append(batch_data.numpy())
        return tf.concat(originals, axis=0).numpy(), tf.concat(reconstructions, axis=0).numpy()
