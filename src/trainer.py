import tensorflow as tf
from tqdm import tqdm
from .losses import encoder_loss, discriminator_loss
from .masking import generate_masked_views, mix_features
import numpy as np



class ACAETrainer:
    def __init__(self, projection_head, encoder, decoder, discriminator, config):
        # Models
        self.projection_head = projection_head
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

        # Hyperparameters from config (with safe defaults)
        self.latent_dim = config.get("latent_dim", 256)
        self.lambda_d = config.get("lambda_d", 1.0)
        self.lambda_e = config.get("lambda_e", 1.0)
        self.recon_weight = config.get("recon_weight", 1.0)
        self.batch_size = config.get("batch_size", 128)
        self.lr = config.get("lr", 1e-4)
        self.mask_rates = config.get("mask_rates", [0.05, 0.15, 0.3, 0.5])

        # Optimizers (same as paper: Adam + weight decay)
        self.enc_dec_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr, weight_decay=1e-4
        )
        self.disc_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr, weight_decay=1e-4
        )

        # Logs
        self.loss_log = {"recon": [], "disc": [], "enc": []}

        # Metrics to avoid per‑step .numpy() syncs
        self.recon_metric = tf.keras.metrics.Mean(name="recon_mean")
        self.disc_metric = tf.keras.metrics.Mean(name="disc_mean")
        self.enc_metric = tf.keras.metrics.Mean(name="enc_mean")

    @tf.function(jit_compile=True)
    def _train_step(self, batch):
        batch = tf.cast(batch, tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            # 1. Projection: (B, 64, F) -> (B, 32, 128)
            projected_batch = self.projection_head(batch, training=True)

            # 2. Encode & Reconstruct
            z = self.encoder(projected_batch, training=True)
            x_hat = self.decoder(z, training=True)
            recon_loss = tf.reduce_mean(tf.square(batch - x_hat))

            # 3. Masked views (positive samples) in projection space
            masked_views = generate_masked_views(projected_batch, self.mask_rates)
            pos_latents = [self.encoder(view, training=True) for view in masked_views]

            # --- POSITIVE PAIRS: [z, z_mixed(z, z_pos)] ---
            z_pos_all = []
            alpha_all = []
            for z_pos in pos_latents:
                z_mixed, alpha = mix_features(z, z_pos)  # (B, latent_dim)
                z_pair = tf.concat([z, z_mixed], axis=1)  # (B, 2*latent_dim)
                z_pos_all.append(z_pair)
                alpha_all.append(alpha)

            z_mixed_pos = tf.concat(z_pos_all, axis=0)  # (B*num_views, 2*latent_dim)
            alpha_pos = tf.concat(alpha_all, axis=0)    # (B*num_views, 1)

            # --- NEGATIVE PAIRS: [z, z_mixed(z, shuffled z)] ---
            z_neg_mixed, beta_neg = mix_features(z, tf.random.shuffle(z))
            z_neg_pair = tf.concat([z, z_neg_mixed], axis=1)

            # 5. Discriminator outputs
            d_out_pos = self.discriminator(z_mixed_pos, training=True)
            d_out_neg = self.discriminator(z_neg_pair, training=True)

            # 6. Losses (unchanged, same as paper)
            loss_disc = discriminator_loss(d_out_pos, d_out_neg, alpha_pos, beta_neg)
            loss_enc = encoder_loss(d_out_pos, d_out_neg, beta_neg)

            total_loss = (
                self.recon_weight * recon_loss
                + self.lambda_d * loss_disc
                + self.lambda_e * loss_enc
            )

        # Backprop: encoder + decoder + projection_head
        enc_dec_vars = (
            self.projection_head.trainable_variables
            + self.encoder.trainable_variables
            + self.decoder.trainable_variables
        )
        disc_vars = self.discriminator.trainable_variables

        enc_dec_grads = tape.gradient(total_loss, enc_dec_vars)
        disc_grads = tape.gradient(loss_disc, disc_vars)

        self.enc_dec_optimizer.apply_gradients(zip(enc_dec_grads, enc_dec_vars))
        self.disc_optimizer.apply_gradients(zip(disc_grads, disc_vars))

        return recon_loss, loss_disc, loss_enc

    def fit(self, train_ds, val_ds=None, epochs=50):
        best_val_loss = float("inf")
        patience = 5
        wait = 0

        for epoch in range(epochs):
            # Reset metrics each epoch
            self.recon_metric.reset_state()
            self.disc_metric.reset_state()
            self.enc_metric.reset_state()

            print(f"\n🌟 Epoch {epoch + 1}/{epochs}")
            bar = tqdm(train_ds, desc="Training", unit="batch")

            for batch in bar:
                r_loss, d_loss, e_loss = self._train_step(batch)

                # Update metrics on‑device
                self.recon_metric.update_state(r_loss)
                self.disc_metric.update_state(d_loss)
                self.enc_metric.update_state(e_loss)

                bar.set_postfix(
                    {
                        "Recon": f"{self.recon_metric.result().numpy():.4f}",
                        "Disc": f"{self.disc_metric.result().numpy():.4f}",
                        "Enc": f"{self.enc_metric.result().numpy():.4f}",
                    }
                )

            avg_recon = float(self.recon_metric.result().numpy())
            avg_disc = float(self.disc_metric.result().numpy())
            avg_enc = float(self.enc_metric.result().numpy())

            # ---- compute validation reconstruction loss ----
            if val_ds is not None:
                val_losses = []
                for vbatch in val_ds:
                    vbatch = tf.cast(vbatch, tf.float32)
                    proj = self.projection_head(vbatch, training=False)
                    z = self.encoder(proj, training=False)
                    x_hat = self.decoder(z, training=False)
                    v_loss = tf.reduce_mean(tf.square(vbatch - x_hat))
                    val_losses.append(v_loss.numpy())
                val_recon = float(np.mean(val_losses))
            else:
                val_recon = avg_recon  # fallback if no val_ds

            print(
                f"✅ Epoch {epoch + 1} Done — "
                f"Train Recon: {avg_recon:.4f} | Val Recon: {val_recon:.4f} | "
                f"Disc: {avg_disc:.4f} | Enc: {avg_enc:.4f}"
            )
            self.loss_log["recon"].append(avg_recon)
            self.loss_log["disc"].append(avg_disc)
            self.loss_log["enc"].append(avg_enc)

            # ---- early stopping on validation recon loss ----
            if val_recon < best_val_loss:
                best_val_loss = val_recon
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(
                        f"🛑 Early stopping at epoch {epoch + 1} — "
                        f"no val recon improvement."
                    )
                    break

    def reconstruct(self, dataset):
        reconstructions = []
        originals = []
        for batch in dataset:
            batch = tf.cast(batch, tf.float32)
            proj = self.projection_head(batch, training=False)
            z = self.encoder(proj, training=False)
            x_hat = self.decoder(z, training=False)
            reconstructions.append(x_hat.numpy())
            originals.append(batch.numpy())
        return (
            tf.concat(originals, axis=0).numpy(),
            tf.concat(reconstructions, axis=0).numpy(),
        )

    def get_reconstruction_errors(self, dataset, y_true):
        scores = []
        for batch in dataset:
            batch = tf.cast(batch, tf.float32)
            proj = self.projection_head(batch, training=False)
            z = self.encoder(proj, training=False)
            x_hat = self.decoder(z, training=False)
            mse = tf.reduce_mean(tf.square(batch - x_hat), axis=[1, 2])
            scores.extend(mse.numpy())
        return np.array(scores), np.array(y_true)


