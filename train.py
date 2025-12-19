import yaml
import argparse
import os
import csv
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras import backend as K
import tensorflow as tf

from src.trainer import ACAETrainer
from src.models import (
    build_encoder,
    build_decoder,
    build_discriminator,
    build_projection_head,
)
from src.utils import load_smd_windows, build_tf_datasets
from src.metrics import compute_auc, compute_fc1, compute_pa_k_auc


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_best_f1(scores, labels, step_size=200):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)

    min_score, max_score = np.min(scores), np.max(scores)
    if min_score == max_score:
        return 0.0, 0.0, 0.0, max_score

    thresholds = np.linspace(min_score, max_score, step_size)

    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_thresh = thresholds[0]

    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_thresh = thresh

    return best_f1, best_precision, best_recall, best_thresh


def train_on_machine(machine_id, config):
    print(f"\n🚀 Training on {machine_id}...")

    # 1. Load Data (Point-wise Evaluation Mode)
    train_w, test_w, test_labels_pointwise, _, _ = load_smd_windows(
        data_root="/home/utsab/Downloads/smd/ServerMachineDataset",
        machine_id=machine_id,
        window=64,
        train_stride=2,
    )

    train_ds, val_ds, test_ds = build_tf_datasets(
        train_w,
        test_w,
        val_split=0.2,
        batch_size=config["batch_size"],
    )

    # 2. Build Models
    features = train_w.shape[-1]  # 38 for SMD

    projection_head = build_projection_head(input_shape=(64, features))
    encoder = build_encoder(
        input_shape_projected=(32, 128),
        latent_dim=config["latent_dim"],
    )
    decoder = build_decoder(
        latent_dim=config["latent_dim"],
        output_steps=64,
        output_features=features,
    )
    discriminator = build_discriminator(latent_dim=config["latent_dim"])

    # Try to resume from checkpoints if they exist
    machine_dir = f"checkpoints/{machine_id}"
    if os.path.isdir(machine_dir):
        try:
            projection_head.load_weights(f"{machine_dir}/projection.weights.h5")
            encoder.load_weights(f"{machine_dir}/encoder.weights.h5")
            decoder.load_weights(f"{machine_dir}/decoder.weights.h5")
            discriminator.load_weights(f"{machine_dir}/discriminator.weights.h5")
            print(f"🔁 Resumed weights from {machine_dir}")
        except Exception as e:
            print(f"⚠️ Could not load checkpoints for {machine_id}: {e}")
            import traceback
            traceback.print_exc()

    # 3. Trainer
    trainer = ACAETrainer(
        projection_head=projection_head,
        encoder=encoder,
        decoder=decoder,
        discriminator=discriminator,
        config=config,
    )
    trainer.fit(train_ds, val_ds=val_ds, epochs=config["epochs"])

    # 4. Evaluation (Point-wise reconstruction errors)
    _, reconstructions = trainer.reconstruct(test_ds)
    recons_flat = reconstructions.reshape(-1, features)
    originals_flat = test_w.reshape(-1, features)

    point_scores = np.sum(np.square(originals_flat - recons_flat), axis=1)

    # Align with labels length
    min_len = min(len(point_scores), len(test_labels_pointwise))
    point_scores = point_scores[:min_len]
    test_labels_pointwise = test_labels_pointwise[:min_len]

    # ---- METRICS ----
    auc = compute_auc(point_scores, test_labels_pointwise)
    best_f1, prec_pt, rec_pt, thr = get_best_f1(point_scores, test_labels_pointwise)
    preds = (point_scores > thr).astype(int)
    fc1 = compute_fc1(preds, test_labels_pointwise)
    pa_auc, ks, f1s_pa = compute_pa_k_auc(
        point_scores,
        test_labels_pointwise,
        base_threshold=thr,
        k_values=np.linspace(0.1, 1.0, 10),
    )

    print(
        f"✅ {machine_id} -> "
        f"AUC: {auc:.4f} | "
        f"Fc1: {fc1:.4f} | "
        f"Best F1: {best_f1:.4f} (P={prec_pt:.4f}, R={rec_pt:.4f}, thr={thr:.5f}) | "
        f"PA%K-AUC: {pa_auc:.4f}"
    )

    # 5. Save Checkpoints
    os.makedirs(machine_dir, exist_ok=True)
    try:
        projection_head.save_weights(f"{machine_dir}/projection.weights.h5")
        encoder.save_weights(f"{machine_dir}/encoder.weights.h5")
        decoder.save_weights(f"{machine_dir}/decoder.weights.h5")
        discriminator.save_weights(f"{machine_dir}/discriminator.weights.h5")
    except Exception as e:
        print(f"⚠️ Could not save checkpoints for {machine_id}: {e}")

    # 6. Clean up TF/VRAM to avoid leaks
    del trainer, projection_head, encoder, decoder, discriminator
    K.clear_session()
    import gc
    gc.collect()

    return auc, fc1, pa_auc



def run_all_machines(config):
    # Hardcoded: start from machine-1-6
    machine_ids = [f"machine-3-{i}" for i in range(10, 12)]  # 6, 7, 8
    skip_machines = set([])

    os.makedirs("results", exist_ok=True)
    csv_path = "results/smd_results.csv"

    # Append to existing CSV (do not overwrite)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["machine_id", "auc", "fc1", "pa_k_auc"])

    results = []
    for mid in machine_ids:
        if mid in skip_machines:
            continue
        try:
            auc, fc1, pa_auc = train_on_machine(mid, config)
            results.append((mid, auc, fc1, pa_auc))

            # Append to CSV immediately
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([mid, auc, fc1, pa_auc])
        except Exception as e:
            print(f"❌ Failed on {mid}: {e}")
            import traceback
            traceback.print_exc()

    print("\n📊 Final Scores Across All Machines:")
    print(f"{'Machine':15s} | {'AUC':8s} | {'Fc1':8s} | {'PA%K-AUC':10s}")
    for mid, auc, fc1, pa_auc in results:
        print(f"{mid:15s} | {auc:.4f} | {fc1:.4f} | {pa_auc:.4f}")

    print(f"\n💾 Results saved to: {csv_path}")


def main(config_path, run_all=False):
    config = load_config(config_path)

    if run_all:
        run_all_machines(config)
    else:
        train_on_machine("machine-1-1", config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ACAE on SMD")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--all", action="store_true", help="Train on all SMD machines"
    )
    args = parser.parse_args()

    main(args.config, run_all=args.all)

