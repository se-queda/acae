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
from src.utils import build_tf_datasets
from src.metrics import compute_auc, compute_fc1, compute_pa_k_auc
from src.data_loaders.psmloader import load_psm_windows
from src.data_loaders.smdloader import load_smd_windows
from src.data_loaders.smaploader import load_smap_windows
from src.data_loaders.msloader import load_msl_windows


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
    dataset_type = config.get("dataset", "SMD").upper()
    print(f"\n🚀 Dual-Anchor Engine Deployment [{dataset_type}]: {machine_id}")

    # Standardized Parameters
    W = config["window_size"] 
    BS = config["batch_size"] 
    VS = config["val_split"]  
    EP = config["epochs"]     
    L = config["latent_dim"]  
    stride = config["stride"]
    # 1. Dynamic Loader Selection 
    data_root = config.get("data_root")
    if dataset_type == "PSM":
        train_final, test_final, test_labels, _, _ = load_psm_windows(data_root, config)
    elif dataset_type == "SMAP":
        train_final, test_final, test_labels, _, _ = load_smap_windows(data_root, machine_id, config)
    elif dataset_type == "MSL":
        train_final, test_final, test_labels, _, _ = load_msl_windows(data_root, machine_id, config)
    else: 
        train_final, test_final, test_labels, _, _ = load_smd_windows(data_root, machine_id, config)
    
    topo = train_final['topology']
    phy_dim = len(topo.idx_phy)
    res_dim = len(topo.idx_res)
    
    # 🚀 DYNAMIC PATIENCE SELECTION
    if phy_dim == 0:
        # Extreme Machine (e.g., G-7): No Physics consensus
        # We need high patience to overcome the 'Val MSE: nan' trap
        current_patience = 100 
        print(f"📡 Mode: Residual-Only. Setting High Patience ({current_patience})")
    else:
        # Healthy Machine (e.g., A-1): Has Physical consensus
        # Low patience prevents overfitting on deterministic telemetry
        current_patience = config.get("patience", 10)
        print(f"🛰️ Mode: Dual-Anchor. Setting Standard Patience ({current_patience})")

    config['patience'] = current_patience
    # 2. Build Datasets
    from src.utils import build_tf_datasets
    train_ds, val_ds, _ = build_tf_datasets(train_final, test_final, val_split=VS, batch_size=BS)

    # 3. Build Models
    from src.models import build_dual_encoder, build_dual_decoder, build_discriminator
    encoder = build_dual_encoder(input_shape_sys=(W, phy_dim), input_shape_res=(W, res_dim), config=config)
    decoder = build_dual_decoder(feat_sys=phy_dim, feat_res=res_dim, output_steps=W, config=config)
    discriminator = build_discriminator(input_dim=L * 2)

    # 4. Trainer Initialization
    from src.trainer import DualAnchorACAETrainer
    trainer = DualAnchorACAETrainer(encoder=encoder, decoder=decoder, discriminator=discriminator, config=config, topology=topo)
    
    # Training with Early Stopping
    trainer.fit(train_ds, val_ds=val_ds, epochs=EP)
    
    # Save Weights
    save_path = f"results/weights/{dataset_type}/{machine_id}"
    os.makedirs(save_path, exist_ok=True)
    trainer.encoder.save_weights(f"{save_path}/encoder.weights.h5")
    trainer.decoder.save_weights(f"{save_path}/decoder.weights.h5")
# 5. Scoring & Diagnostic Normalization
    recons = trainer.reconstruct(test_final) 
    # Paper-aligned sequence length logic 
    actual_len = (recons['res_orig'].shape[0] - 1) * stride + W

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
        windowed_labels = test_labels[:recons['res_orig'].shape[0] * W].reshape(-1, W)
        test_labels = aggregate_scores(windowed_labels, stride, W, actual_len, is_label=True)

    # 1. Physics Branch [cite: 486, 487]
    if phy_dim > 0:
        se_p = np.mean(np.square(recons['phy_orig'] - recons['phy_hat']), axis=-1)
        e_p = aggregate_scores(se_p, stride, W, actual_len)
    else:
        e_p = np.zeros(actual_len)

    # 2. Lone Wolf Branch
    res_orig, res_hat = recons['res_orig'], recons['res_hat']
    if len(topo.res_to_lone_local) > 0:
        se_l = np.mean(np.square(res_orig[:, :, topo.res_to_lone_local] - res_hat[:, :, topo.res_to_lone_local]), axis=-1)
        e_l = aggregate_scores(se_l, stride, W, actual_len)
    else:
        e_l = np.zeros(actual_len)

    # 3. Dead Sentinel Branch
    if len(topo.res_to_dead_local) > 0:
        se_d = np.mean(np.square(res_orig[:, :, topo.res_to_dead_local] - 0.0), axis=-1)
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

def run_all_entities(config):
    # 1. Retrieve values explicitly from the config passed by main()
    dataset_type = config.get("dataset", "SMD").upper()
    data_root = config.get("data_root")
    
    # 🔍 SAFETY CHECK: Ensure the path isn't None before joining
    if data_root is None:
        raise ValueError(f"❌ Data root for {dataset_type} is None. Check your config.yaml paths.")

    # --- 2. ID Discovery ---
    if dataset_type == "PSM":
        machine_ids = ["PSM_Pooled"]
    else:
        # Determine file extension based on dataset [cite: 533]
        ext = ".txt" if dataset_type == "SMD" else ".npy"
        train_path = os.path.join(data_root, "train")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Could not find train folder at {train_path}")
            
        # Get all IDs (filenames without extension)
        machine_ids = sorted([f.replace(ext, "") for f in os.listdir(train_path) if f.endswith(ext)])
    
    print(f"📂 Found {len(machine_ids)} entities for {dataset_type}")

    os.makedirs("results", exist_ok=True)
    csv_path = f"results/final_{dataset_type}_baseline.csv"

    # Initialize CSV with headers if it doesn't exist [cite: 508, 512, 515]
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "auc", "fc1", "pa_k_auc", "f1_std", "f1_pa", "precision", "recall"])

    # Sequential execution for GPU stability and early stopping [cite: 545]
    for mid in machine_ids:
        p = mp.Process(target=worker_task, args=(mid, config, csv_path))
        p.start()
        p.join()


def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # 1. Load the base config (this contains the 'datasets' block)
    base_config = load_config(args.config)
    
    # Target datasets in order
    target_datasets = ["PSM", "SMD", "MSL", "SMAP"]
    print(f"\n📡 Dual-Anchor Engine Benchmarking Suite Initiated")
    for ds_name in target_datasets:
        print(f"\n{'='*60}\n📡 BENCHMARKING: {ds_name}\n{'='*60}")

        # 2. Create a fresh copy for this specific dataset
        config = base_config.copy()
        config["dataset"] = ds_name
        
        # 3. CRITICAL: Pull the path from the nested YAML structure [cite: 533]
        # This maps base_config['datasets'][ds_name]['data_root'] -> config['data_root']
        try:
            config["data_root"] = base_config["datasets"][ds_name]["data_root"]
        except KeyError:
            print(f"❌ Error: Dataset {ds_name} or its data_root not found in config.yaml")
            continue
        
        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)

        if ds_name == "PSM":
            # PSM is typically a single file execution [cite: 533]
            mid = "PSM_Pooled" # or your specific PSM filename
            csv_path = f"results/final_{ds_name}_baseline.csv"
            p = mp.Process(target=worker_task, args=(mid, config, csv_path))
            p.start()
            p.join()
        else:
            # Iterates through the directories for SMD, MSL, and SMAP [cite: 533]
            run_all_entities(config)

if __name__ == "__main__":
    main()