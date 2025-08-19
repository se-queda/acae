import yaml
import argparse
import os
import csv
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.keras import backend as K

from src.trainer import ACAETrainer
from src.models import build_encoder, build_decoder, build_discriminator
from src.utils import load_smd_windows, build_tf_datasets


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train_on_machine(machine_id, config):
    print(f"\n🚀 Training on {machine_id}...")
    train_w, test_w, y_test_win, _, _ = load_smd_windows(
        data_root="data/smd",
        machine_id=machine_id,
        window=64,
        stride=2
    )

    train_ds, val_ds, test_ds = build_tf_datasets(
        train_w, test_w,
        val_split=0.2,
        batch_size=config['batch_size']
    )

    encoder = build_encoder(input_shape=(64, train_w.shape[-1]), latent_dim=config['latent_dim'])
    decoder = build_decoder(latent_dim=config['latent_dim'], output_shape=(64, train_w.shape[-1]))
    discriminator = build_discriminator(latent_dim=config['latent_dim'])

    trainer = ACAETrainer(encoder, decoder, discriminator, config)
    trainer.fit(train_ds, val_ds=val_ds, epochs=config['epochs'])

    scores, y_true = trainer.get_reconstruction_errors(test_ds, y_test_win)
    auc = roc_auc_score(y_true, scores)
    print(f"✅ {machine_id} AUC: {auc:.4f}")

    # Create a subfolder per machine
    machine_dir = f"checkpoints/{machine_id}"
    os.makedirs(machine_dir, exist_ok=True)

    encoder.save_weights(f"{machine_dir}/encoder.weights.h5")
    decoder.save_weights(f"{machine_dir}/decoder.weights.h5")
    discriminator.save_weights(f"{machine_dir}/discriminator.weights.h5")

    # 🧹 Clean up TensorFlow graph to free memory
    K.clear_session()

    return auc

def run_all_machines(config):
    machine_ids = (
            [f"machine-1-{i}" for i in range(1, 8)] +
            [f"machine-2-{i}" for i in range(1, 12)] +
            [f"machine-3-{i}" for i in range(1, 11)]
    )
    skip_machines = set([
        "machine-1-1", "machine-1-2", "machine-1-3", "machine-1-4", "machine-1-5","machine-1-6","machine-1-7",
        "machine-2-1", "machine-2-2", "machine-2-3", "machine-2-4", "machine-2-5", "machine-2-6", "machine-2-7", "machine-2-8",
        "machine-2-9", "machine-2-10", "machine-2-11", "machine-2-12"
    ])
    results = []
    for mid in machine_ids:
        if mid in skip_machines:
            continue
        auc = train_on_machine(mid, config)
        results.append((mid, auc))

    print("\n📊 Final AUC Scores Across All Machines:")
    for mid, auc in results:
        print(f"{mid:15s} : AUC = {auc:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/smd_auc_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["machine_id", "auc_score"])
        writer.writerows(results)

    print("\n💾 Results saved to: results/smd_auc_results.csv")


def main(config_path, run_all=False):
    config = load_config(config_path)

    if run_all:
        run_all_machines(config)
    else:
        train_on_machine("machine-1-1", config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ACAE on smd")
    parser.add_argument('--config', type=str, default="config.yaml", help='Path to config file')
    parser.add_argument('--all', action='store_true', help='Train on all 28 smd machines')
    args = parser.parse_args()

    main(args.config, run_all=args.all)
