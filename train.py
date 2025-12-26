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

from src.trainer import DualAnchorACAETrainer
from src.models import build_dual_encoder, build_dual_decoder, build_discriminator
from src.utils import load_smd_windows, build_tf_datasets
from src.metrics import compute_auc, compute_fc1, compute_pa_k_auc

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_best_metrics(scores, labels, step_size=100):
    """
    Optimizes independently for Fc1 and PA%K AUC.
    Returns only the best scores achieved for each.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    min_score, max_score = np.min(scores), np.max(scores)
    
    if min_score == max_score:
        return 0.0, 0.0

    thresholds = np.linspace(min_score, max_score, step_size)
    
    best_fc1 = 0.0
    best_pa_auc = 0.0

    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        
        # 1. Optimize for Fc1 (Event-wise Recall + Time-wise Precision)
        # This is the actual metric reported in the paper 
        current_fc1 = compute_fc1(preds, labels)
        if current_fc1 > best_fc1:
            best_fc1 = current_fc1
            
        # 2. Optimize for PA%K AUC (Integration of adjusted F1s)
        # Calculated across K=0.0 to 1.0 per paper guidelines 
        current_pa_auc, _, _ = compute_pa_k_auc(scores, labels, base_threshold=thresh)
        if current_pa_auc > best_pa_auc:
            best_pa_auc = current_pa_auc

    return best_fc1, best_pa_auc

def train_on_machine(machine_id, config):
    print(f"\n🚀 K-SHIELD Engine Deployment: {machine_id}")

    # 1. Load Data
    train_final, test_final, test_labels, _, _ = load_smd_windows(
        data_root=config.get("data_root", "/home/utsab/Downloads/smd/ServerMachineDataset"),
        machine_id=machine_id,
        window=64,
        train_stride=config.get("stride", 1)
    )
    
    topo = train_final['topology']
    phy_dim = len(topo.idx_phy)
    res_dim = len(topo.idx_res)

    # 2. Build Datasets
    train_ds, val_ds, test_ds = build_tf_datasets(
        train_final, test_final, 
        val_split=0.2, 
        batch_size=config["batch_size"]
    )

    # 3. Build Models
    encoder = build_dual_encoder(input_shape_sys=(64, phy_dim), input_shape_res=(64, res_dim))
    decoder = build_dual_decoder(feat_sys=phy_dim, feat_res=res_dim)
    discriminator = build_discriminator(latent_dim=config["latent_dim"])

    # 4. Trainer (Alpha VPO lives in the _train_step)
    trainer = DualAnchorACAETrainer(
        encoder=encoder, decoder=decoder, discriminator=discriminator,
        config=config, topology=topo
    )
    
    trainer.fit(train_ds, val_ds=val_ds, epochs=config["epochs"])
    
    save_path = f"results/weights/{machine_id}"
    os.makedirs(save_path, exist_ok=True)
    
    trainer.encoder.save_weights(f"{save_path}/encoder.weights.h5")
    trainer.decoder.save_weights(f"{save_path}/decoder.weights.h5")
    trainer.discriminator.save_weights(f"{save_path}/discriminator.weights.h5")
    print(f"💾 All 4 weights saved to {save_path}")

    # 5. Balanced Anomaly Scoring
    recons = trainer.reconstruct(test_final)
    
    # --- Branch 1: Physics (HNN Path) ---
    e_p = np.mean(np.square(recons['phy_orig'] - recons['phy_hat']), axis=-1)
    
    # --- Branch 2: Residual (Lone Wolves) ---
    res_orig, res_hat = recons['res_orig'], recons['res_hat']
    e_l = np.mean(np.square(
        res_orig[:, :, topo.res_to_lone_local] - res_hat[:, :, topo.res_to_lone_local]
    ), axis=-1)
    
    # --- Branch 3: Sentinel (Dead Sensors) ---
    e_d = np.mean(np.square(res_orig[:, :, topo.res_to_dead_local] - 0.0), axis=-1)
    
    # --- THE KEY INNOVATION: MIN-MAX NORMALIZATION ---
    # This prevents err_dead (the bully) from silencing the Physics branch
    def norm(s):
        return (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-6)

    # Combine normalized signals: Each branch now has an equal vote
    # We still give Sentinel a slight nudge (2.0) because dead sensors are high-confidence
    score_p = norm(e_p)
    score_l = norm(e_l)
    score_d = norm(e_d)
    
    combined_window_scores = score_p + score_l + (1.0 * score_d)
    point_scores = combined_window_scores.flatten() 

    # 6. Alignment & Metric Optimization
    min_len = min(len(point_scores), len(test_labels))
    point_scores, test_labels = point_scores[:min_len], test_labels[:min_len]

    fc1_score, pa_auc_score = get_best_metrics(point_scores, test_labels)
    auc_score = compute_auc(point_scores, test_labels)

    print(f"✅ {machine_id} -> AUC: {auc_score:.4f} | Fc1: {fc1_score:.4f} | PA%K: {pa_auc_score:.4f}")

    K.clear_session()
    gc.collect()

    return auc_score, fc1_score, pa_auc_score
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
    csv_path = "results/acae_hnn_sobolev_consensus_duaLatent.csv"

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