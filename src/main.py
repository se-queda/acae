import yaml
import argparse
import os
import csv
import numpy as np
import gc
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
import traceback

from src.trainer import DualAnchorACAETrainer
from src.models import DualEncoder, DualDecoder, Discriminator
from src.utils import plot_reconstruction
from src.dataset_builder import dataloader
from src.metrics import compute_auc, compute_fc1, compute_pa_k_auc
from src.data_loaders.psmloader import load_psm_windows
from src.data_loaders.smdloader import load_smd_windows

from src.configs.fno_config import fno_config
from src.configs.hnn_config import hnn_config
from src.configs.global_config import global_config




def get_best_metrics(scores, labels, step_size=100):
    
    #Finds optimal thresholds for all evaluation philosophies. 
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    min_score, max_score = np.min(scores), np.max(scores)
    
    if min_score == max_score:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    thresholds = np.linspace(min_score, max_score, step_size)
    
    best_fc1 = 0.0
    best_pa_auc = 0.0
    best_f1_std = 0.0      
    best_f1_pa = 0.0       
    best_prec_std = 0.0
    best_recall_std = 0.0
    best_qeds = 0.0   

    from src.metrics import (
        _point_adjustment_at_k,
        _f1_from_preds,
        compute_qeds         
    )
    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        
        # 1. Fc1
        current_fc1 = compute_fc1(preds, labels)
        if current_fc1 > best_fc1:
            best_fc1 = current_fc1
            
        current_pa_auc, _, _ = compute_pa_k_auc(scores, labels, base_threshold=thresh)
        if current_pa_auc > best_pa_auc:
            best_pa_auc = current_pa_auc
            
        # 2. Standard Point-wise
        prec, rec, f1_std, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0
        )
        if f1_std > best_f1_std:
            best_f1_std = f1_std
            best_prec_std = prec
            best_recall_std = rec

        #Point-Adjusted 
        adjusted_preds = _point_adjustment_at_k(preds, labels, K=0.0001) 
        f1_pa = _f1_from_preds(adjusted_preds, labels)
        if f1_pa > best_f1_pa:
            best_f1_pa = f1_pa

        #Early Detection (
        current_qeds = compute_qeds(preds, labels)
        if current_qeds > best_qeds:
            best_qeds = current_qeds

    return (best_fc1,best_pa_auc,best_f1_std,best_f1_pa,best_prec_std,best_recall_std,best_qeds)


def train_on_machine(machine_id, config):
    dataset_type = config.get("dataset", "SMD").upper()
    print(f"\n🚀 Dual-Anchor Engine Deployment [{dataset_type}]: {machine_id}")

    # Standardized Parameters
    W = global_config["window_size"]
    BS = global_config["batch_size"]
    VS = global_config["val_split"]
    EP = global_config["epochs"]
    stride = global_config["stride"]
    hnn_dim = hnn_config["hnn_dim"]
    fno_dim = fno_config["fno_dim"]
    L = hnn_dim

    # 1. Dynamic Loader Selection 
    data_root = config.get("data_root")
    if dataset_type == "PSM":
        train_final, test_final, test_labels = load_psm_windows(data_root, config)
    else:
        train_final, test_final, test_labels = load_smd_windows(data_root, machine_id, config)
    
    
    topo = train_final['topology']
    phy_dim = len(topo.idx_phy)
    res_dim = len(topo.idx_res)
    
    # 🚀 DYNAMIC PATIENCE SELECTION
    if phy_dim == 0:
        # Extreme Machine : No Consensus sensor
        # We need high patience to overcome the 'Val MSE: nan' trap
        current_patience = 100 
        print(f"📡 Mode: Residual-Only. Setting High Patience ({current_patience})")
    else:
        # Healthy Machine : Has Physical consensus
        # Low patience prevents overfitting on deterministic telemetry
        current_patience = global_config.get("patience", 10)
        print(f"🛰️ Mode: Dual-Anchor. Setting Standard Patience ({current_patience})")

    global_config["patience"] = current_patience
    # 2. Build Datasets
    train_ds, val_ds, _ = dataloader(train_final, test_final, val_split=VS, batch_size=BS)

    # 3. Trainer Initialization
    trainer = DualAnchorACAETrainer(
        encoder=DualEncoder,
        decoder=DualDecoder,
        discriminator=Discriminator,
        hnn_config=hnn_config,
        fno_config=fno_config,
        global_config=global_config,
        topology=topo
    )
    
    # Training with Early Stopping
    trainer.fit(train_ds, val_ds=val_ds, epochs=EP)

    # Save Weights (PyTorch) + load check
    save_path = f"results/weights/{dataset_type}/{machine_id}"
    os.makedirs(save_path, exist_ok=True)
    encoder_path = os.path.join(save_path, "encoder.pt")
    decoder_path = os.path.join(save_path, "decoder.pt")
    torch.save(trainer.encoder.state_dict(), encoder_path)
    torch.save(trainer.decoder.state_dict(), decoder_path)
    trainer.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"), strict=True)
    trainer.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"), strict=True)

    # 5. Scoring & Diagnostic Normalization
    recons = trainer.reconstruct(test_final, batch_size=BS)
    #  sequence length logic 
    num_windows = test_final['res'].shape[0]
    actual_len = (num_windows - 1) * stride + W

    def aggregate_scores(windowed_data, stride, window_size, total_len, is_label=False):
        """Averages overlapping windows into point-wise sequence per Eq 9."""
        # Convert NaNs to zero to prevent signal death [cite: 101, 104]
        windowed_data = np.nan_to_num(windowed_data, nan=0.0)
        scores, counts = np.zeros(total_len), np.zeros(total_len)
        
        for i, window_val in enumerate(windowed_data):
            start, end = i * stride, i * stride + window_size
            actual_end = min(end, total_len)
            slice_len = actual_end - start
            if slice_len > 0:
                if is_label:
                    # For labels, if a point is anomalous in ANY window, it's an anomaly [cite: 512, 513]
                    scores[start:actual_end] = np.maximum(scores[start:actual_end], window_val[:slice_len])
                else:
                    scores[start:actual_end] += window_val[:slice_len]
                    counts[start:actual_end] += 1
        
        return scores if is_label else scores / np.maximum(counts, 1)

    # --- CRITICAL LABEL REALIGNMENT ---
    # Most NASA loaders return flattened (N*W) labels. We must linearize them.
    if test_labels.shape[0] > actual_len:
        print(f"🔄 [REALIGN] Converting {len(test_labels)} windowed labels to {actual_len} linear steps...")
        windowed_labels = test_labels[:num_windows * W].reshape(-1, W)
        test_labels = aggregate_scores(windowed_labels, stride, W, actual_len, is_label=True)

    # 1. Physics Branch 
    if phy_dim > 0:
        se_p = recons['se_p']
        e_p = aggregate_scores(se_p, stride, W, actual_len)
    else:
        e_p = np.zeros(actual_len)

    # 2. Lone Wolf Branch
    if len(topo.res_to_lone_local) > 0:
        se_l = recons['se_l']
        e_l = aggregate_scores(se_l, stride, W, actual_len)
    else:
        e_l = np.zeros(actual_len)

    # 3. Dead Sentinel Branch
    if len(topo.res_to_dead_local) > 0:
        se_d = recons['se_d']
        e_d = aggregate_scores(se_d, stride, W, actual_len)
    else:
        e_d = np.zeros(actual_len)

    def norm(s): 
        s_min, s_max = np.nanmin(s), np.nanmax(s)
        if s_max - s_min < 1e-9: return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min + 1e-6)

    # Composite Point-wise anomaly scores [cite: 231, 559]
    point_scores = (norm(e_p) + norm(e_l) + norm(e_d))

    # 6. Metrics Calculation [cite: 508, 509, 512, 515]
    test_labels = test_labels[:actual_len]
    point_scores = point_scores[:len(test_labels)]
    
    # Debug Sync Check
    print(f"🏁 [FINAL SYNC] Scores: {len(point_scores)} | Labels: {len(test_labels)} | Anomaly Rate: {np.mean(test_labels):.2%}")

    auc_score = compute_auc(point_scores, test_labels) 
    fc1, pa_auc, f1_std, f1_pa, prec, rec = get_best_metrics(point_scores, test_labels)
    # ... [Rest of returns]
    print(f"🏆 [RESULTS] AUC: {auc_score:.4f} | F1-Score: {fc1:.4f} | PA-AUC: {pa_auc:.4f} | F1-Std: {f1_std:.4f} | F1-PA: {f1_pa:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}" )
    return auc_score, fc1, pa_auc, f1_std, f1_pa, prec, rec

 
def worker_task(mid, config, csv_path):
    """Child process executor with clean TF initialization."""
    try:
        # GPU Setup
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        # Run Training & Scoring
        auc, fc1, pa_auc, f1_std, f1_pa, prec, rec = train_on_machine(mid, config)
        
        # Save to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([mid, auc, fc1, pa_auc, f1_std, f1_pa, prec, rec])
            
    except Exception as e:
        print(f"❌ Failed on {mid}: {e}")
        traceback.print_exc()

def run_all_entities(config):
    dataset_type = config.get("dataset", "SMD").upper()
    data_root = config.get("data_root")
    
    # --- 1. ID Discovery ---
    if dataset_type == "PSM":
        machine_ids = ["PSM_Pooled"]
    else:
        # Determine file extension based on dataset
        ext = ".txt" if dataset_type == "SMD" else ".npy"
        train_path = os.path.join(data_root, "train")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Could not find train folder at {train_path}")
            
        # Get all IDs (filenames without extension)
        machine_ids = sorted([f.replace(ext, "") for f in os.listdir(train_path) if f.endswith(ext)])
    
    print(f"📂 Found {len(machine_ids)} entities for {dataset_type}")

    os.makedirs("results", exist_ok=True)
    csv_path = f"results/final_{dataset_type}_baseline.csv"

    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "auc", "fc1", "pa_k_auc", "f1_std", "f1_pa", "precision", "recall"])

    # Sequential execution for GPU stability
    for mid in machine_ids:
        p = mp.Process(target=worker_task, args=(mid, config, csv_path))
        p.start()
        p.join()

def main():
    # Set the multiprocessing start method globally before any TF imports
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--all", action="store_true")
    # Optional override for quick testing
    parser.add_argument("--id", type=str, help="Specify a single machine/channel ID to test")
    args = parser.parse_args()

    # Assuming load_config is available in your environment
    config = load_config(args.config)
    dataset_type = config.get("dataset", "SMD").upper()
    
    if args.all:
        run_all_entities(config)
    else:
        # Logic for a single-point test run
        if args.id:
            mid = args.id
        else:
            # Smart defaults for single runs based on dataset
            defaults = {"SMD": "machine-3-1", "MSL": "T-10", "SMAP": "P-1", "PSM": "PSM_Pooled"}
            mid = defaults.get(dataset_type, "test_entity")

        print(f"🧪 Starting single-entity test run: {mid}")
        csv_path = f"results/single_test_{dataset_type}.csv"
        
        p = mp.Process(target=worker_task, args=(mid, config, csv_path))
        p.start()
        p.join()

if __name__ == "__main__":
    main()
