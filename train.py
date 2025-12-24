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
    print(f"\n🚀 Starting Training: {machine_id}")

    # 1. Load Data
    train_w, test_w, test_labels_pointwise, _, _ = load_smd_windows(
        data_root=config.get("data_root", "/home/utsab/Downloads/smd/ServerMachineDataset"),
        machine_id=machine_id,
        window=64,
        train_stride=2,
    )

    train_ds, val_ds, test_ds = build_tf_datasets(
        train_w, test_w, val_split=0.2, batch_size=config["batch_size"],
    )

    # 2. Build Models (Fresh instances per machine)
    features = train_w.shape[-1]
    projection_head = build_projection_head(input_shape=(64, features))
    encoder = build_encoder(input_shape_projected=(32, 128), latent_dim=config["latent_dim"])
    decoder = build_decoder(latent_dim=config["latent_dim"], output_steps=64, output_features=features)
    discriminator = build_discriminator(latent_dim=config["latent_dim"])

    # 3. Trainer
    trainer = ACAETrainer(
        projection_head=projection_head,
        encoder=encoder,
        decoder=decoder,
        discriminator=discriminator,
        config=config,
    )
    
    trainer.fit(train_ds, val_ds=val_ds, epochs=config["epochs"])

    # 4. Evaluation
    _, reconstructions = trainer.reconstruct(test_ds)
    recons_flat = reconstructions.reshape(-1, features)
    originals_flat = test_w.reshape(-1, features)
    point_scores = np.sum(np.square(originals_flat - recons_flat), axis=1)

    min_len = min(len(point_scores), len(test_labels_pointwise))
    point_scores, test_labels_pointwise = point_scores[:min_len], test_labels_pointwise[:min_len]

    auc = compute_auc(point_scores, test_labels_pointwise)
    best_f1, prec_pt, rec_pt, thr = get_best_f1(point_scores, test_labels_pointwise)
    preds = (point_scores > thr).astype(int)
    fc1 = compute_fc1(preds, test_labels_pointwise)
    pa_auc, _, _ = compute_pa_k_auc(point_scores, test_labels_pointwise, base_threshold=thr)

    print(f"✅ {machine_id} -> AUC: {auc:.4f} | Fc1: {fc1:.4f} | PA%K: {pa_auc:.4f}")

    # --- 5. AGGRESSIVE MEMORY CLEANUP ---
    del train_w, test_w, train_ds, val_ds, test_ds
    del trainer, projection_head, encoder, decoder, discriminator
    K.clear_session()
    gc.collect()

    return auc, fc1, pa_auc

def run_all_machines(config):
    machine_ids = [
    # Group 1
    "machine-1-1", "machine-1-2", "machine-1-3", "machine-1-4", 
    "machine-1-5", "machine-1-6", "machine-1-7", "machine-1-8",
    
    # Group 2
    "machine-2-1", "machine-2-2", "machine-2-3", "machine-2-4", 
    "machine-2-5", "machine-2-6", "machine-2-7", "machine-2-8", "machine-2-9",
    
    # Group 3
    "machine-3-1", "machine-3-2", "machine-3-3", "machine-3-4", 
    "machine-3-5", "machine-3-6", "machine-3-7", "machine-3-8", 
    "machine-3-9", "machine-3-10", "machine-3-11"
]
    
    os.makedirs("results", exist_ok=True)
    csv_path = "results/acae_jerk_smd_consensusMask.csv"

    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["machine_id", "auc", "fc1", "pa_k_auc"])

    for mid in machine_ids:
        # Spawn fresh process per machine to isolate RAM
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
        # Default single machine test
        mid = "machine-1-1"
        auc, fc1, pa_auc = train_on_machine(mid, config)
        
        os.makedirs("results", exist_ok=True)
        csv_path = "results/mini_expriment_2.csv"
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["machine_id", "auc", "fc1", "pa_k_auc"])
            writer.writerow([mid, auc, fc1, pa_auc])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    main(args.config, run_all=args.all)
