import tensorflow as tf
from tqdm import tqdm
from .losses import encoder_loss, discriminator_loss
from .masking import mix_features
import numpy as np

class ACAETrainer:
    def __init__(self, projection_head, encoder, decoder, discriminator, config):
        self.projection_head = projection_head
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

        self.latent_dim = config.get("latent_dim", 256)
        self.lambda_d = config.get("lambda_d", 1.0)
        self.lambda_e = config.get("lambda_e", 1.0)
        self.recon_weight = config.get("recon_weight", 1.0)
        self.lr = config.get("lr", 1e-4)

        self.enc_dec_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, weight_decay=1e-4)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, weight_decay=1e-4)

        self.loss_log = {"recon": [], "disc": [], "enc": []}
        self.recon_metric = tf.keras.metrics.Mean(name="recon_mean")
        self.disc_metric = tf.keras.metrics.Mean(name="disc_mean")
        self.enc_metric = tf.keras.metrics.Mean(name="enc_mean")

    @tf.function(jit_compile=True)
    def _train_step(self, packed_batch):
        packed_batch = tf.cast(packed_batch, tf.float32)
        anchor = packed_batch[:, 0, :, :]        
        physics_views = packed_batch[:, 1:, :, :] 

        with tf.GradientTape(persistent=True) as tape:
            z = self.encoder(self.projection_head(anchor, training=True), training=True)
            x_hat = self.decoder(z, training=True)
            recon_loss = tf.reduce_mean(tf.square(anchor - x_hat))

            z_pos_all, alpha_all = [], []
            for i in range(4):
                view = physics_views[:, i, :, :]
                z_aug = self.encoder(self.projection_head(view, training=True), training=True)
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
            total_loss = (self.recon_weight * recon_loss + self.lambda_d * loss_disc + self.lambda_e * loss_enc)

        enc_dec_vars = self.projection_head.trainable_variables + self.encoder.trainable_variables + self.decoder.trainable_variables
        self.enc_dec_optimizer.apply_gradients(zip(tape.gradient(total_loss, enc_dec_vars), enc_dec_vars))
        self.disc_optimizer.apply_gradients(zip(tape.gradient(loss_disc, self.discriminator.trainable_variables), self.discriminator.trainable_variables))
        return recon_loss, loss_disc, loss_enc

    def fit(self, train_ds, val_ds=None, epochs=50):
        best_val_loss = float("inf")
        patience, wait = 5, 0
        for epoch in range(epochs):
            self.recon_metric.reset_state()
            self.disc_metric.reset_state()
            self.enc_metric.reset_state()
            print(f"\n🌟 Epoch {epoch + 1}/{epochs}")
            bar = tqdm(train_ds, desc="Training", unit="batch")
            for batch in bar:
                r_loss, d_loss, e_loss = self._train_step(batch)
                self.recon_metric.update_state(r_loss)
                self.disc_metric.update_state(d_loss)
                self.enc_metric.update_state(e_loss)
                bar.set_postfix({"Recon": f"{r_loss:.4f}", "Disc": f"{d_loss:.4f}", "Enc": f"{e_loss:.4f}"})

            if val_ds is not None:
                val_losses = []
                for v_batch in val_ds:
                    v_anchor = tf.cast(v_batch[:, 0, :, :], tf.float32) # Val is packed
                    z_val = self.encoder(self.projection_head(v_anchor, training=False), training=False)
                    v_hat = self.decoder(z_val, training=False)
                    val_losses.append(tf.reduce_mean(tf.square(v_anchor - v_hat)).numpy())
                val_recon = float(np.mean(val_losses))
            else: val_recon = self.recon_metric.result().numpy()

            print(f"✅ Epoch {epoch+1} | Val Recon: {val_recon:.4f}")
            if val_recon < best_val_loss:
                best_val_loss, wait = val_recon, 0
            else:
                wait += 1
                if wait >= patience: break

    # --- RESTORED FUNCTIONS ---

    def reconstruct(self, dataset):
        """
        Restored for test-time evaluation. Handles standard 3D test batches.
        """
        reconstructions, originals = [], []
        for batch in dataset:
            # Test data is NOT packed, it's (B, T, F)
            batch = tf.cast(batch, tf.float32)
            z = self.encoder(self.projection_head(batch, training=False), training=False)
            x_hat = self.decoder(z, training=False)
            reconstructions.append(x_hat.numpy())
            originals.append(batch.numpy())
        return (
            tf.concat(originals, axis=0).numpy(),
            tf.concat(reconstructions, axis=0).numpy(),
        )

    def get_reconstruction_errors(self, dataset, y_true):
        """
        Restored for point-wise scoring.
        """
        scores = []
        for batch in dataset:
            batch = tf.cast(batch, tf.float32)
            z = self.encoder(self.projection_head(batch, training=False), training=False)
            x_hat = self.decoder(z, training=False)
            # MSE per window
            mse = tf.reduce_mean(tf.square(batch - x_hat), axis=[1, 2])
            scores.extend(mse.numpy())
        return np.array(scores), np.array(y_true)
