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
import traceback

from src.trainer import DualAnchorACAETrainer
from src.models import build_dual_encoder, build_dual_decoder, build_discriminator
from src.utils import load_smd_windows, build_tf_datasets
from src.metrics import compute_auc, compute_fc1, compute_pa_k_auc

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_best_metrics(scores, labels, step_size=100):
    """
    Finds optimal thresholds for all evaluation philosophies. [cite: 489, 516]
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    min_score, max_score = np.min(scores), np.max(scores)
    
    if min_score == max_score:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    thresholds = np.linspace(min_score, max_score, step_size)
    
    best_fc1 = 0.0
    best_pa_auc = 0.0
    best_f1_std = 0.0      
    best_f1_pa = 0.0       
    best_prec_std = 0.0
    best_recall_std = 0.0

    from src.metrics import _point_adjustment_at_k, _f1_from_preds

    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        
        # 1. Paper Specific Metrics (Strict) [cite: 511, 514]
        current_fc1 = compute_fc1(preds, labels)
        if current_fc1 > best_fc1:
            best_fc1 = current_fc1
            
        current_pa_auc, _, _ = compute_pa_k_auc(scores, labels, base_threshold=thresh)
        if current_pa_auc > best_pa_auc:
            best_pa_auc = current_pa_auc
            
        # 2. Standard Point-wise (No Adjustment)
        prec, rec, f1_std, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0
        )
        if f1_std > best_f1_std:
            best_f1_std = f1_std
            best_prec_std = prec
            best_recall_std = rec

        # 3. Classic SOTA Point-Adjusted (Generous) [cite: 515]
        adjusted_preds = _point_adjustment_at_k(preds, labels, K=0.0001) 
        f1_pa = _f1_from_preds(adjusted_preds, labels)
        if f1_pa > best_f1_pa:
            best_f1_pa = f1_pa

    return best_fc1, best_pa_auc, best_f1_std, best_f1_pa, best_prec_std, best_recall_std

def train_on_machine(machine_id, config):
    print(f"\n🚀 Dual-Anchor Engine Deployment: {machine_id}")

    # --- Structural Initialization from YAML ---
    W = config["window_size"]  # [cite: 544, 580]
    S = config["stride"]       # [cite: 544, 577]
    BS = config["batch_size"]  # 
    VS = config["val_split"]   # 
    EP = config["epochs"]      # 
    L = config["latent_dim"]   # [cite: 543]

    # 1. Load Data (Dynamic Dimensions)
    train_final, test_final, test_labels, _, _ = load_smd_windows(
        data_root=config.get("data_root", "/home/utsab/Downloads/smd/ServerMachineDataset"),
        machine_id=machine_id,
        config=config # Pass whole config for savgol/sparsity params
    )
    
    topo = train_final['topology']
    phy_dim = len(topo.idx_phy)
    res_dim = len(topo.idx_res)

    # 2. Build Datasets using config BS and VS
    train_ds, val_ds, test_ds = build_tf_datasets(
        train_final, test_final, 
        val_split=VS, 
        batch_size=BS
    )

    # 3. Build Models using config L and h_dim
    encoder = build_dual_encoder(
        input_shape_sys=(W, phy_dim), 
        input_shape_res=(W, res_dim), 
        config=config
    )
    decoder = build_dual_decoder(
        feat_sys=phy_dim, 
        feat_res=res_dim, 
        output_steps=W, 
        config=config
    )
    
    # Discriminator: Anchor (L) + Candidate (L) = L*2 
    discriminator = build_discriminator(input_dim=L * 2)

    # 4. Trainer Initialization (Loss weights λ1, λ2, and alpha_vpo) [cite: 475]
    trainer = DualAnchorACAETrainer(
        encoder=encoder, decoder=decoder, discriminator=discriminator,
        config=config, topology=topo
    )
    
    trainer.fit(train_ds, val_ds=val_ds, epochs=EP)
    
    save_path = f"results/weights/{machine_id}"
    os.makedirs(save_path, exist_ok=True)
    
    trainer.encoder.save_weights(f"{save_path}/encoder.weights.h5")
    trainer.decoder.save_weights(f"{save_path}/decoder.weights.h5")
    trainer.discriminator.save_weights(f"{save_path}/discriminator.weights.h5")

    # 5. Scoring Logic (Point-wise Error calculation) [cite: 485, 486]
    recons = trainer.reconstruct(test_final)

    # Physics Branch Error (with configured alpha_vpo)
    e_p = np.mean(np.square(recons['phy_orig'] - recons['phy_hat']), axis=-1)
    res_orig, res_hat = recons['res_orig'], recons['res_hat']

    # Lone Wolf vs Sentinel specific errors
    e_l = np.mean(np.square(res_orig[:, :, topo.res_to_lone_local] - res_hat[:, :, topo.res_to_lone_local]), axis=-1)
    e_d = np.mean(np.square(res_orig[:, :, topo.res_to_dead_local] - 0.0), axis=-1)

    def norm(s):
        return (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-6)

    score_p, score_l, score_d = norm(e_p), norm(e_l), norm(e_d)
    point_scores = (score_p + score_l + score_d).flatten()


    # 6. Metrics and Thresholding [cite: 509, 511, 514]
    min_len = min(len(point_scores), len(test_labels))
    point_scores, test_labels = point_scores[:min_len], test_labels[:min_len]


    fc1, pa_auc, f1_std, f1_pa, prec, rec = get_best_metrics(point_scores, test_labels)
    auc_score = compute_auc(point_scores, test_labels)


    K.clear_session()
    gc.collect()


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

def run_all_machines(config):
    machine_ids = [
        "machine-3-6", "machine-3-7", "machine-3-8", 
        "machine-3-9", "machine-3-10", "machine-3-11"
    ]
    
    os.makedirs("results", exist_ok=True)
    csv_path = "results/acae_hnn_sobolev_TCN_consensus_duaLatent.csv"

    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["machine_id", "auc", "fc1", "pa_k_auc", "f1_std", "f1_pa", "precision", "recall"])

    # Sequential execution via individual processes for maximum GPU memory safety
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
    args = parser.parse_args()

    config = load_config(args.config)
    
    if args.all:
        run_all_machines(config)
    else:
        # Single machine test run
        mid = "machine-2-1"
        # We still use a separate process even for one machine to maintain isolation
        csv_path = "results/single_machine_test.csv"
        p = mp.Process(target=worker_task, args=(mid, config, csv_path))
        p.start()
        p.join()

if __name__ == "__main__":
    main()