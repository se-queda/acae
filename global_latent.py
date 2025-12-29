import os
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from src.models import build_dual_encoder, build_dual_decoder, build_discriminator
from src.trainer import DualAnchorACAETrainer
from src.router import MachineTopology

# --- 1. SMD DATA LOADER ---
def load_smd_raw(machine_id, data_root, window=64):
    """
    Loads raw SMD telemetry and labels from text files.
    """
    test_path = os.path.join(data_root, "test", f"{machine_id}.txt")
    label_path = os.path.join(data_root, "test_label", f"{machine_id}.txt")
    
    # Load SMD .txt (standardized 38-dimensional telemetry)
    test_data = np.loadtxt(test_path, delimiter=',').astype(np.float32)
    test_labels = np.loadtxt(label_path, delimiter=',').astype(np.int32)
    return test_data, test_labels

# --- 2. TRAINER LOADER WITH TOPOLOGY SYNC ---
def get_trainer_fixed(machine_id, data_root, config, p_dim=19):
    """
    Dynamically discovers the correct p_dim/r_dim split used during training.
    """
    W = config["window_size"]
    total_dim = 38 # SMD Constant Dimension
    r_dim = total_dim - p_dim

    test_raw, test_labels = load_smd_raw(machine_id, data_root, W)

    # 1. Initialize shell model components
    topo = MachineTopology(list(range(p_dim)), list(range(p_dim, total_dim)), [], [])
    encoder = build_dual_encoder((W, p_dim), (W, r_dim), config)
    decoder = build_dual_decoder(p_dim, r_dim, W, config)
    discriminator = build_discriminator(input_dim=config["latent_dim"] * 2)
    trainer = DualAnchorACAETrainer(encoder, decoder, discriminator, config, topo)
    
    weights_path = f"results/weights/SMD/{machine_id}/encoder.weights.h5"
    
    try:
        # ACAE utilizes trained weights for test-stage latent extraction
        trainer.encoder.load_weights(weights_path)
    except ValueError as e:
        # Regex extraction of the actual physical dimension from Keras shape mismatch
        msg = str(e)
        match = re.search(r"Received: value\.shape=\((\d+),", msg)
        if match:
            real_p = int(match.group(1))
            real_r = total_dim - real_p
            print(f"🎯 {machine_id} Topology Sync: {real_p} Phy / {real_r} Res")
            tf.keras.backend.clear_session()
            return get_trainer_fixed(machine_id, data_root, config, p_dim=real_p)
        else:
            raise e

    # 2. Windowing with verified topology split
    X_p = np.array([test_raw[i : i + W, :p_dim] for i in range(len(test_raw) - W + 1)])
    X_r = np.array([test_raw[i : i + W, p_dim:] for i in range(len(test_raw) - W + 1)])
    L = np.array([test_labels[i : i + W] for i in range(len(test_labels) - W + 1)])

    # Window is labeled anomalous if ANY point within it is an anomaly
    return trainer, X_p, X_r, np.any(L > 0, axis=1)

# --- 3. MANIFOLD EXECUTION ---
def run_global_manifold(machine_list, data_root, config):
    global_z_sys, global_z_res, global_labels, machine_tags = [], [], [], []

    print(f"🌐 Aggregating {len(machine_list)} SMD machines...")

    for mid in machine_list:
        try:
            trainer, X_p, X_r, L_win = get_trainer_fixed(mid, data_root, config)
            
            # Sampling for clean visualization
            idx = np.random.choice(len(X_p), min(500, len(X_p)), replace=False)
            
            # Extract decoupled latent variables
            zs, zr, _ = trainer.encoder([X_p[idx], X_r[idx]], training=False)
            
            global_z_sys.append(zs.numpy())
            global_z_res.append(zr.numpy())
            global_labels.append(L_win[idx])
            machine_tags.extend([mid] * len(idx))
            
            tf.keras.backend.clear_session()
            print(f"✅ {mid} added.")
        except Exception as e:
            print(f"❌ Failed {mid}: {e}")

    # Aggregation for Global Fleet Analysis
    Z_S, Z_R = np.concatenate(global_z_sys), np.concatenate(global_z_res)
    LBL, TAGS = np.concatenate(global_labels), np.array(machine_tags)

    print(f"🪄 Computing t-SNE for {len(Z_S)} points...")
    tsne = TSNE(n_components=2, perplexity=35, random_state=42, init='random')
    z_s_2d = tsne.fit_transform(Z_S)
    z_r_2d = tsne.fit_transform(Z_R)

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
    # Consensus Manifold: Visualizing Systemic operational states
    sns.scatterplot(x=z_s_2d[~LBL,0], y=z_s_2d[~LBL,1], hue=TAGS[~LBL], ax=ax1, palette="tab20", alpha=0.5, s=15)
    ax1.scatter(z_s_2d[LBL, 0], z_s_2d[LBL, 1], c='red', marker='x', s=30, label="Anomaly", alpha=0.8)
    ax1.set_title("Global SMD Consensus Manifold (HNN)\nTarget: Systemic Cluster Identity")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='x-small')

    # Residual Manifold: Visualizing semantic anomaly separation
    sns.scatterplot(x=z_r_2d[~LBL,0], y=z_r_2d[~LBL,1], hue=TAGS[~LBL], ax=ax2, palette="tab20", alpha=0.5, s=15, legend=False)
    ax2.scatter(z_r_2d[LBL, 0], z_r_2d[LBL, 1], c='red', marker='x', s=30, alpha=0.8)
    ax2.set_title("Global SMD Residual Manifold (TCN)\nTarget: Stochastic Detail Separation")

    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/global_SMD_manifold.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    smd_root = "/home/utsab/Downloads/smd/ServerMachineDataset"
    # Analyzing both Group 1 and Group 2 machines
    machines = [f"machine-1-{i}" for i in range(1, 9)] + [f"machine-2-{i}" for i in range(1, 9)]
    
    config = {
        "window_size": 64, "latent_dim": 512, "hnn_feature_dim": 256,
        "hnn_steps": 3, "hnn_dt": 0.1, "lr": 1e-4, "patience": 5, 
        "batch_size": 128, "dropout": 0.1, "recon_weight": 10.0,
        "lambda_d": 1.0, "lambda_e": 1.0, "alpha_vpo": 2.0, "sentinel_weight": 5.0
    }
    run_global_manifold(machines, smd_root, config)










