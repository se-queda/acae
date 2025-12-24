import tensorflow as tf
from tqdm import tqdm
from .losses import encoder_loss, discriminator_loss
from .masking import mix_features
import numpy as np

# Use namespace as per your preference
namespace = tf.keras

class ACAETrainer:
    def __init__(self, encoder, decoder, discriminator, config):
        """
        Initializes trainer using parameters from config.yaml.
        """
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

        # Model Hyperparameters from Config
        self.latent_dim = config.get("latent_dim", 256) #
        self.lr = config.get("lr", 1e-4) #
        
        # Loss Weights
        self.lambda_d = config.get("lambda_d", 1.0)
        self.lambda_e = config.get("lambda_e", 1.0)
        self.recon_weight = config.get("recon_weight", 1.0)
        self.alpha_vpo = config.get("alpha_vpo", 10.0)

        # Physics/HNN Specific (New for Baseline 5)
        self.hnn_steps = config.get("hnn_steps", 3)
        self.hnn_dt = config.get("hnn_dt", 0.1)

        # Optimized Optimizer for HNN Stability
        # clipnorm is critical to handle Hamiltonian gradient spikes
        self.enc_dec_optimizer = namespace.optimizers.Adam(
            learning_rate=self.lr, 
            clipnorm=1.0, 
            weight_decay=1e-4
        )
        self.disc_optimizer = namespace.optimizers.Adam(
            learning_rate=self.lr, 
            clipnorm=1.0, 
            weight_decay=1e-4
        )

        # Metrics
        self.recon_metric = namespace.metrics.Mean(name="recon_mean")
        self.disc_metric = namespace.metrics.Mean(name="disc_mean")
        self.enc_metric = namespace.metrics.Mean(name="enc_mean")

    @tf.function(jit_compile=True)
    def _train_step(self, packed_batch):
        packed_batch = tf.cast(packed_batch, tf.float32)
        # anchor: (B, 64, 22), physics_views: (B, 4, 64, 22)
        anchor = packed_batch[:, 0, :, :]        
        physics_views = packed_batch[:, 1:5, :, :] # 4 augmented views
        jerk_w = packed_batch[:, 5, :, :] # Jerk feature for VPO weighting
        
        with tf.GradientTape(persistent=True) as tape:
            # Encoder handles internal Projection + HNN Leapfrog
            z = self.encoder(anchor, training=True)
            x_hat = self.decoder(z, training=True)
            
            # Weighted Reconstruction (VPO Logic)
            j_abs = tf.abs(jerk_w)
            j_thresh = tf.reduce_mean(j_abs) + tf.math.reduce_std(j_abs)
            vpo_weights = tf.where(j_abs > j_thresh, self.alpha_vpo, 1.0)
            pointwise_error = tf.square(anchor - x_hat)
            recon_loss = tf.reduce_mean(pointwise_error * vpo_weights)

            # Contrastive Task: Feature Combination
            z_pos_all, alpha_all = [], []
            for i in range(4):
                view = physics_views[:, i, :, :]
                z_aug = self.encoder(view, training=True)
                z_mixed, alpha = mix_features(z, z_aug)
                z_pos_all.append(tf.concat([z, z_mixed], axis=1))
                alpha_all.append(alpha)

            z_mixed_pos = tf.concat(z_pos_all, axis=0)
            alpha_pos = tf.concat(alpha_all, axis=0)
            
            # Negative Sampling
            z_neg_mixed, beta_neg = mix_features(z, tf.random.shuffle(z))
            z_neg_pair = tf.concat([z, z_neg_mixed], axis=1)

            # Adversarial Decomposition
            d_out_pos = self.discriminator(z_mixed_pos, training=True)
            d_out_neg = self.discriminator(z_neg_pair, training=True)

            loss_disc = discriminator_loss(d_out_pos, d_out_neg, alpha_pos, beta_neg)
            loss_enc = encoder_loss(d_out_pos, d_out_neg, beta_neg)
            
            # Total Loss sum
            total_loss = (self.recon_weight * recon_loss + 
                          self.lambda_d * loss_disc + 
                          self.lambda_e * loss_enc)

        # Apply Gradients
        enc_dec_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        self.enc_dec_optimizer.apply_gradients(
            zip(tape.gradient(total_loss, enc_dec_vars), enc_dec_vars)
        )
        self.disc_optimizer.apply_gradients(
            zip(tape.gradient(loss_disc, self.discriminator.trainable_variables), 
                self.discriminator.trainable_variables)
        )
        
        return recon_loss, loss_disc, loss_enc

    def fit(self, train_ds, val_ds=None, epochs=200): #
        best_val_loss = float("inf")
        patience = 5 # As per paper early stopping
        wait = 0 
        
        for epoch in range(epochs):
            self.recon_metric.reset_state()
            self.disc_metric.reset_state()
            self.enc_metric.reset_state()
            
            bar = tqdm(train_ds, desc=f"Epoch {epoch+1}", unit="batch")
            for batch in bar:
                r_loss, d_loss, e_loss = self._train_step(batch)
                self.recon_metric.update_state(r_loss)
                self.disc_metric.update_state(d_loss)
                self.enc_metric.update_state(e_loss)
                bar.set_postfix({
                    "Recon": f"{r_loss:.4f}", 
                    "Disc": f"{d_loss:.4f}", 
                    "Enc": f"{e_loss:.4f}"
                })

            # Validation stage
            if val_ds is not None:
                val_losses = []
                for v_batch in val_ds:
                    v_anchor = tf.cast(v_batch[:, 0, :, :], tf.float32)
                    z_val = self.encoder(v_anchor, training=False)
                    v_hat = self.decoder(z_val, training=False)
                    val_losses.append(tf.reduce_mean(tf.square(v_anchor - v_hat)).numpy())
                val_recon = float(np.mean(val_losses))
            else: 
                val_recon = self.recon_metric.result().numpy()

            print(f"✅ Val Recon: {val_recon:.4f}")
            
            # Early stopping check
            if val_recon < best_val_loss:
                best_val_loss, wait = val_recon, 0
            else:
                wait += 1
                if wait >= patience: 
                    print("Early stopping triggered.")
                    break

    def reconstruct(self, dataset):
        reconstructions, originals = [], []
        for batch in dataset:
            batch = tf.cast(batch, tf.float32)
            z = self.encoder(batch, training=False)
            x_hat = self.decoder(z, training=False)
            reconstructions.append(x_hat.numpy())
            originals.append(batch.numpy())
        return tf.concat(originals, axis=0).numpy(), tf.concat(reconstructions, axis=0).numpy()

    def get_reconstruction_errors(self, dataset, y_true):
        """
        Point-wise error calculation for anomaly scoring.
        """
        scores = []
        for batch in dataset:
            batch = tf.cast(batch, tf.float32)
            z = self.encoder(batch, training=False)
            x_hat = self.decoder(z, training=False)
            mse = tf.reduce_mean(tf.square(batch - x_hat), axis=[1, 2])
            scores.extend(mse.numpy())
        return np.array(scores), np.array(y_true)