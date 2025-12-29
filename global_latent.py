import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
from sklearn.manifold import TSNE
import tensorflow as tf
import seaborn as sns
from src.models import build_dual_encoder, build_dual_decoder, build_discriminator
from src.trainer import DualAnchorACAETrainer
from src.router import route_features # Required for topology matching
from src.router import MachineTopology
# --- UPDATED NASA HELPER FUNCTIONS ---

def load_nasa_test_data(machine_id, data_root, window_size=128):
    """
    Parses NASA .npy telemetry and CSV labels.
    """
    test_path = os.path.join(data_root, "test", f"{machine_id}.npy")
    csv_path = os.path.join(data_root, "labeled_anomalies.csv")
    
    # Load Telemetry
    test_data = np.load(test_path).astype(np.float32)
    
    # Load and Parse Labels
    df = pd.read_csv(csv_path)
    machine_info = df[df['chan_id'] == machine_id].iloc[0]
    test_labels = np.zeros(len(test_data), dtype=np.int32)
    anomaly_indices = ast.literal_eval(machine_info['anomaly_sequences'])
    for start, end in anomaly_indices:
        test_labels[start : end + 1] = 1
    
    # Windowing
    X, L = [], []
    for i in range(0, len(test_data) - window_size + 1, 1):
        X.append(test_data[i : i + window_size])
        L.append(test_labels[i : i + window_size])
    
    return np.array(X), np.array(L)

def get_trained_trainer_for_nasa(machine_id, data_root, config):
    W = config["window_size"]
    dataset_type = config.get("dataset", "SMAP").upper()
    total_dim = 25 if dataset_type == "SMAP" else 55 
    
    # 1. Load Data
    X_raw, L_raw = load_nasa_test_data(machine_id, data_root, W)
    
    # 2. Topology Correction Loop
    def build_and_load(p_dim, r_dim):
        # Create topology with correct class name
        # We pass empty lists for lone/dead as they aren't needed for latent projection
        topo = MachineTopology(
            idx_phy=list(range(p_dim)), 
            idx_res=list(range(p_dim, total_dim)),
            idx_lone=[], 
            idx_dead=[]
        )

        encoder = build_dual_encoder((W, p_dim), (W, r_dim), config)
        decoder = build_dual_decoder(p_dim, r_dim, W, config)
        discriminator = build_discriminator(input_dim=config["latent_dim"] * 2)
        trainer = DualAnchorACAETrainer(encoder, decoder, discriminator, config, topo)
        
        weights_path = f"results/weights/{dataset_type}/{machine_id}/encoder.weights.h5"
        
        try:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Missing: {weights_path}")
            
            trainer.encoder.load_weights(weights_path)
            return trainer, p_dim, r_dim
        except (ValueError, tf.errors.InvalidArgumentError) as e:
            msg = str(e)
            # Find the saved residual dimension (X) from the error message
            match = re.search(r"Received: value\.shape=\((\d+),", msg)
            if match:
                actual_res_dim = int(match.group(1))
                actual_phy_dim = total_dim - actual_res_dim
                print(f"🔄 Topology Correction for {machine_id}: Rebuilding as {actual_phy_dim} Phy / {actual_res_dim} Res")
                tf.keras.backend.clear_session()
                return build_and_load(actual_phy_dim, actual_res_dim)
            else:
                raise e

    # Initial routing logic
    try:
        (t_phy, t_res, _, _), _, _ = route_features(X_raw[0], X_raw[0])
        initial_p, initial_r = t_phy.shape[-1], t_res.shape[-1]
    except:
        initial_p, initial_r = total_dim // 2, total_dim - (total_dim // 2)

    trainer, final_p, final_r = build_and_load(initial_p, initial_r)

    data_dict = {
        'phy': np.expand_dims(X_raw[:, :, :final_p], axis=1),
        'res': X_raw[:, :, final_p:],
        'labels': L_raw
    }
    return trainer, data_dict

# --- GLOBAL PROJECTION SCRIPT ---

def run_global_manifold(machine_list, data_root, config):
    global_z_sys, global_z_res = [], []
    global_labels, machine_tags = [], []

    print(f"🌐 Aggregating {len(machine_list)} {config['dataset']} machines...")

    for mid in machine_list:
        try:
            trainer, data_dict = get_trained_trainer_for_nasa(mid, data_root, config)
            
            total_windows = len(data_dict['phy'])
            # Sample 500 windows per machine to avoid T-SNE memory crash
            sample_size = min(500, total_windows)
            idx = np.random.choice(total_windows, sample_size, replace=False)
            
            p_mini = tf.cast(data_dict['phy'][idx, 0, :, :], tf.float32)
            r_mini = tf.cast(data_dict['res'][idx], tf.float32)
            
            zs, zr, _ = trainer.encoder([p_mini, r_mini], training=False)
            
            global_z_sys.append(zs.numpy())
            global_z_res.append(zr.numpy())
            global_labels.append(np.any(data_dict['labels'][idx] > 0, axis=1))
            machine_tags.extend([mid] * sample_size)
            
            tf.keras.backend.clear_session()
        except Exception as e:
            print(f"❌ Skipping {mid} due to error: {e}")

    # Process and Plot
    Z_S = np.concatenate(global_z_sys, axis=0)
    Z_R = np.concatenate(global_z_res, axis=0)
    LBL = np.concatenate(global_labels, axis=0)
    TAGS = np.array(machine_tags)

    print(f"🪄 Computing t-SNE for {len(Z_S)} total points...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_jobs=-1)
    z_s_2d = tsne.fit_transform(Z_S)
    z_r_2d = tsne.fit_transform(Z_R)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    sns.scatterplot(x=z_s_2d[~LBL, 0], y=z_s_2d[~LBL, 1], hue=TAGS[~LBL], 
                    ax=ax1, palette="husl", alpha=0.5, s=15, legend='full')
    ax1.scatter(z_s_2d[LBL, 0], z_s_2d[LBL, 1], c='red', marker='x', s=30, label="Anomaly")
    ax1.set_title(f"Global {config['dataset']} HNN Consensus Space")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=3, fontsize='x-small')

    sns.scatterplot(x=z_r_2d[~LBL, 0], y=z_r_2d[~LBL, 1], hue=TAGS[~LBL], 
                    ax=ax2, palette="husl", alpha=0.5, s=15, legend=False)
    ax2.scatter(z_r_2d[LBL, 0], z_r_2d[LBL, 1], c='red', marker='x', s=30)
    ax2.set_title(f"Global {config['dataset']} TCN Residual Space")

    plt.tight_layout()
    plt.savefig(f"results/plots/global_{config['dataset']}_manifold.png")
    plt.show()

if __name__ == "__main__":
    # Discovery of SMAP Machines
    data_root = "/home/utsab/Downloads/SMAP"
    train_dir = os.path.join(data_root, "train")
    
    if os.path.exists(train_dir):
        smap_machines = sorted([f.replace(".npy", "") for f in os.listdir(train_dir) if f.endswith(".npy")])
        print(f"📂 Found {len(smap_machines)} SMAP machines for global manifold.")
    else:
        smap_machines = ["P-1", "S-1", "E-1", "T-1", "P-2", "P-3"]

    # 🚀 FULL SYNC WITH YOUR PROVIDED HYPS
    config = {
        "dataset": "SMAP",
        "data_root": data_root,
        "window_size": 128,         # Window size for ACAE
        "stride": 2,               # Interval 'd'
        "batch_size": 128,          
        "latent_dim": 512,          # z_sys[256] + z_res[256]
        "lr": 0.0001,               
        
        # HNN Hyperparameters
        "hnn_steps": 3,
        "hnn_dt": 0.1,
        "hnn_feature_dim": 256,
        
        # Loss Weights
        "lambda_d": 1.0,          
        "lambda_e": 1.0,          
        "recon_weight": 1.0,      
        "lambda_mom": 0.5,
        "alpha_vpo": 150.0,
        "sentinel_weight": 10.0,
        
        # Preprocessing
        "savgol_len": 11,
        "savgol_poly": 3,
        "sparsity_factor": 10,
        "p_tile": 90,
        "dropout": 0.1,
        
        # Training loop keys (Fixes 'patience' error)
        "epochs": 200,            
        "patience": 8              
    }

    run_global_manifold(smap_machines, config["data_root"], config)