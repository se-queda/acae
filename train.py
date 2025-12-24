import yaml
import argparse
import os
import csv
import numpy as np
import gc
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras import backend as K
import tensorflow as tf
import multiprocessing as mp

# Updated components as per our HNN design
from src.trainer import ACAETrainer
from src.models import (
    build_encoder,
    build_decoder,
    build_discriminator
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
    best_f1, best_precision, best_recall, best_thresh = 0.0, 0.0, 0.0, thresholds[0]

    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_precision, best_recall, best_thresh = f1, precision, recall, thresh

    return best_f1, best_precision, best_recall, best_thresh

def train_on_machine(machine_id, config):
    print(f"\n🚀 Starting High-Res HNN Training: {machine_id}")

    # 1. Load Data
    # Utilizing the full 64-step resolution for HNN stability
    train_w, test_w, test_labels_pointwise, _, _ = load_smd_windows(
        data_root=config.get("data_root", "/home/utsab/Downloads/smd/ServerMachineDataset"),
        machine_id=machine_id,
        window=64,
        train_stride=2, # Balancing coverage and speed
    )

    train_ds, val_ds, test_ds = build_tf_datasets(
        train_w, test_w, val_split=0.2, batch_size=config["batch_size"],
    )

    # 2. Build Models (No more pooling in the Projection Head)
    features = train_w.shape[-1]
    
    # Encoder: Raw Windows (64 steps) -> HNN -> Latent Space
    encoder = build_encoder(input_shape=(64, features), latent_dim=config["latent_dim"])
    
    # Decoder: Mirroring the HNN flow back to sensor space
    decoder = build_decoder(latent_dim=config["latent_dim"], output_steps=64, output_features=features)
    
    # Adversarial Component
    discriminator = build_discriminator(latent_dim=config["latent_dim"])

    # 3. Trainer
    # Driving the Sobolev loss (Position + Momentum)
    trainer = ACAETrainer(
        encoder=encoder,
        decoder=decoder,
        discriminator=discriminator,
        config=config,
    )
    
    trainer.fit(train_ds, val_ds=val_ds, epochs=config["epochs"])

    # 4. Evaluation: Physics-Informed Anomaly Scoring
    _, reconstructions = trainer.reconstruct(test_ds)
    
    # Shape: (N_windows, 64, features)
    originals = test_w 
    
    # Point-wise Euclidean Error (Position)
    mse_error = np.square(originals - reconstructions)
    
    # Momentum Error (Temporal Gradient Match)
    # We calculate the delta between steps to see if the 'flow' is broken
    v_orig = originals[:, 1:, :] - originals[:, :-1, :]
    v_hat = reconstructions[:, 1:, :] - reconstructions[:, :-1, :]
    mom_error = np.square(v_orig - v_hat)
    
    # Combine errors: MSE + 0.5 * Momentum (matching our training objective)
    # We pad the mom_error to match the original window length
    mom_error_padded = np.pad(mom_error, ((0,0), (1,0), (0,0)), mode='edge')
    combined_error = mse_error + (config.get("lambda_mom", 0.5) * mom_error_padded)
    
    # Flatten for point-wise scoring across all sensors
    point_scores = np.sum(combined_error, axis=(1, 2))

    min_len = min(len(point_scores), len(test_labels_pointwise))
    point_scores, test_labels_pointwise = point_scores[:min_len], test_labels_pointwise[:min_len]

    # Metrics
    auc = compute_auc(point_scores, test_labels_pointwise)
    best_f1, prec_pt, rec_pt, thr = get_best_f1(point_scores, test_labels_pointwise)
    preds = (point_scores > thr).astype(int)
    fc1 = compute_fc1(preds, test_labels_pointwise)
    pa_auc, _, _ = compute_pa_k_auc(point_scores, test_labels_pointwise, base_threshold=thr)

    print(f"✅ {machine_id} -> AUC: {auc:.4f} | Fc1: {fc1:.4f} | PA%K: {pa_auc:.4f}")

    # 5. Aggressive Memory Cleanup
    del train_w, test_w, train_ds, val_ds, test_ds
    del trainer, encoder, decoder, discriminator
    K.clear_session()
    gc.collect()

    return auc, fc1, pa_auc

def run_all_machines(config):
    machine_ids = [
        "machine-1-1", "machine-1-2", "machine-1-3", "machine-1-4", 
        "machine-1-5", "machine-1-6", "machine-1-7", "machine-1-8",
        "machine-2-1", "machine-2-2", "machine-2-3", "machine-2-4", 
        "machine-2-5", "machine-2-6", "machine-2-7", "machine-2-8", "machine-2-9",
        "machine-3-1", "machine-3-2", "machine-3-3", "machine-3-4", 
        "machine-3-5", "machine-3-6", "machine-3-7", "machine-3-8", 
        "machine-3-9", "machine-3-10", "machine-3-11"
    ]
    
    os.makedirs("results", exist_ok=True)
    csv_path = "results/acae_hnn_sobolev_consensus.csv"

    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["machine_id", "auc", "fc1", "pa_k_auc"])

    for mid in machine_ids:
        p = mp.Process(target=worker_task, args=(mid, config, csv_path))
        p.start()
        p.join() 

def worker_task(mid, config, csv_path):
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        auc, fc1, pa_auc = train_on_machine(mid, config)
        
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([mid, auc, fc1, pa_auc])
            
    except Exception as e:
        print(f"❌ Failed on {mid}: {e}")

def main(config_path, run_all=False):
    config = load_config(config_path)
    if run_all:
        run_all_machines(config)
    else:
        mid = "machine-1-1"
        train_on_machine(mid, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    main(args.config, run_all=args.all)