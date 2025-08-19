mport yaml
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.keras import backend as K

from src.trainer import ACAETrainer
from src.models import build_encoder, build_decoder, build_discriminator
from src.utils import load_smd_windows, build_tf_datasets


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main(config_path):
    config = load_config(config_path)

    # 📦 Load dataset
    train_w, test_w, y_test_win, _, _ = load_smd_windows(
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

    # 📊 Inference for AUC-ROC
    print("\n🔍 Running post-training inference for AUC-ROC...")
    scores, y_true = trainer.get_reconstruction_errors(test_ds, y_test_win)
    auc = roc_auc_score(y_true, scores)
    print(f"✅ AUC-ROC: {auc:.4f}")

    # 🧹 Optional cleanup
    K.clear_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ACAE on smd")
    parser.add_argument('--config', type=str, default="config.yaml", help='Path to config file')
    args = parser.parse_args()

    main(args.config)

