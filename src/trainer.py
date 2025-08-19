import tensorflow as tf
from tqdm import tqdm
from .losses import encoder_loss, discriminator_loss
from .masking import generate_masked_views, mix_features


class ACAETrainer:
    def __init__(self, encoder, decoder, discriminator, config):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

        # Hyperparameters from config
        self.latent_dim = config.get('latent_dim', 256)
        self.lambda_d = config.get('lambda_d', 1.0)
        self.lambda_e = config.get('lambda_e', 1.0)
        self.recon_weight = config.get('recon_weight', 1.0)
        self.batch_size = config.get('batch_size', 128)
        self.lr = config.get('lr', 1e-4)
        self.mask_rates = config.get('mask_rates', [0.05, 0.15, 0.3, 0.5])

        # Optimizers
        self.enc_dec_optimizer = tf.keras.optimizers.Adam(self.lr)
        self.disc_optimizer = tf.keras.optimizers.Adam(self.lr)

        # Logs
        self.loss_log = {'recon': [], 'disc': [], 'enc': []}

    @tf.function(jit_compile=True)
    def _train_step(self, batch):
        with tf.GradientTape(persistent=True) as tape:
            # Encode & Reconstruct
            z = self.encoder(batch, training=True)
            x_hat = self.decoder(z, training=True)
            recon_loss = tf.reduce_mean(tf.square(batch - x_hat))

            # Masked views → positive samples
            masked_views = generate_masked_views(batch, self.mask_rates)
            pos_latents = [self.encoder(view, training=True) for view in masked_views]

            z_pos_all, alpha_all = [], []
            for z_pos in pos_latents:
                z_mixed, alpha = mix_features(z, z_pos)
                z_pos_all.append(z_mixed)
                alpha_all.append(alpha)

            z_mixed_pos = tf.concat(z_pos_all, axis=0)
            alpha_pos = tf.concat(alpha_all, axis=0)

            # Negative sample
            z_neg, beta_neg = mix_features(z, tf.random.shuffle(z))

            # Discriminator outputs
            d_out_pos = self.discriminator(z_mixed_pos, training=True)
            d_out_neg = self.discriminator(z_neg, training=True)

            # Losses
            loss_disc = discriminator_loss(d_out_pos, d_out_neg, alpha_pos, beta_neg)
            loss_enc = encoder_loss(d_out_pos, d_out_neg, beta_neg)

            total_loss = (
                self.recon_weight * recon_loss +
                self.lambda_d * loss_disc +
                self.lambda_e * loss_enc
            )

        # Backprop
        enc_dec_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        disc_vars = self.discriminator.trainable_variables

        self.enc_dec_optimizer.apply_gradients(zip(
            tape.gradient(total_loss, enc_dec_vars), enc_dec_vars
        ))
        self.disc_optimizer.apply_gradients(zip(
            tape.gradient(loss_disc, disc_vars), disc_vars
        ))

        return recon_loss, loss_disc, loss_enc

    def fit(self, train_ds, val_ds=None, epochs=50):
        best_val_loss = float('inf')
        patience = 5
        wait = 0

        for epoch in range(epochs):
            recon_total, disc_total, enc_total, steps = 0, 0, 0, 0
            print(f"\n🌟 Epoch {epoch + 1}/{epochs}")
            bar = tqdm(train_ds, desc="Training", unit="batch")

            for batch in bar:
                steps += 1
                r_loss, d_loss, e_loss = self._train_step(batch)
                recon_total += r_loss.numpy()
                disc_total += d_loss.numpy()
                enc_total += e_loss.numpy()

                bar.set_postfix({
                    "Recon": f"{recon_total / steps:.4f}",
                    "Disc": f"{disc_total / steps:.4f}",
                    "Enc": f"{enc_total / steps:.4f}"
                })

            avg_recon = recon_total / steps
            avg_disc = disc_total / steps
            avg_enc = enc_total / steps

            print(f"✅ Epoch {epoch+1} Done — Recon: {avg_recon:.4f} | Disc: {avg_disc:.4f} | Enc: {avg_enc:.4f}")
            self.loss_log['recon'].append(avg_recon)
            self.loss_log['disc'].append(avg_disc)
            self.loss_log['enc'].append(avg_enc)

            if avg_recon < best_val_loss:
                best_val_loss = avg_recon
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"🛑 Early stopping at epoch {epoch+1} — no recon improvement.")
                    break

    def reconstruct(self, dataset):
        reconstructions = []
        originals = []
        for batch in dataset:
            z = self.encoder(batch)
            x_hat = self.decoder(z)
            reconstructions.append(x_hat.numpy())
            originals.append(batch.numpy())
        return (
            tf.concat(originals, axis=0).numpy(),
            tf.concat(reconstructions, axis=0).numpy()
        )
    def get_reconstruction_errors(self, dataset, y_true):
        scores = []

        for batch in dataset:
            z = self.encoder(batch, training=False)
            x_hat = self.decoder(z, training=False)
            mse = tf.reduce_mean(tf.square(batch - x_hat), axis=[1, 2])
            scores.extend(mse.numpy())

        return np.array(scores), np.array(y_true)
