import yaml
import argparse
from tensorflow.keras import backend as K

from src.models import build_encoder, build_decoder, build_discriminator
from src.trainer import ACAETrainer
from src.utils import load_smd_windows, build_tf_datasets


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main(config_path):
    # 🧠 Load config
    config = load_config(config_path)

    # 📦 Load data
    train_w, test_w, y_test_win, mean, std = load_smd_windows(
        data_root="data/smd",
        machine_id="machine-1-1",
        window=64,
        stride=2
    )

    train_ds, val_ds, test_ds = build_tf_datasets(
        train_w, test_w,
        val_split=0.2,
        batch_size=config['batch_size']
    )

    # 🏗️ Build models
    encoder = build_encoder(input_shape=(64, train_w.shape[-1]), latent_dim=config['latent_dim'])
    decoder = build_decoder(latent_dim=config['latent_dim'], output_shape=(64, train_w.shape[-1]))
    discriminator = build_discriminator(latent_dim=config['latent_dim'])

    # 🚂 Train ACAE
    trainer = ACAETrainer(encoder, decoder, discriminator, config)
    trainer.fit(train_ds, val_ds=val_ds, epochs=config['epochs'])

    X_orig, X_rec = trainer.reconstruct(test_ds)

    # 🧹 Clear TF session (optional cleanup)
    K.clear_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ACAE on SMD")
    parser.add_argument('--config', type=str, default="config.yaml", help='Path to config file')
    args = parser.parse_args()

    main(args.config)
