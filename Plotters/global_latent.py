import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
import seaborn as sns
from src.models import build_dual_encoder, build_dual_decoder, build_discriminator
from src.trainer import DualAnchorACAETrainer

# --- HELPED FUNCTIONS ---

def load_and_split_data(machine_id, data_root, window_size=64):
    test_path = os.path.join(data_root, "test", f"{machine_id}.txt")
    label_path = os.path.join(data_root, "test_label", f"{machine_id}.txt")
    
    test_data = np.loadtxt(test_path, delimiter=',')
    labels = np.loadtxt(label_path, delimiter=',')
    
    X, L = [], []
    for i in range(0, len(test_data) - window_size + 1, 1):
        X.append(test_data[i : i + window_size])
        L.append(labels[i : i + window_size])
    
    return np.array(X), np.array(L)

def get_trained_trainer_for_machine(machine_id, data_root, config):
    W = config["window_size"]
    X_raw, L_raw = load_and_split_data(machine_id, data_root, W)
    total_dim = X_raw.shape[-1]

    p_guess = 24
    r_guess = total_dim - p_guess

    def build_and_load(p_dim, r_dim):
        encoder = build_dual_encoder((W, p_dim), (W, r_dim), config)
        decoder = build_dual_decoder(p_dim, r_dim, W, config)
        discriminator = build_discriminator(input_dim=config["latent_dim"] * 2)

        trainer = DualAnchorACAETrainer(encoder, decoder, discriminator, config, None)
        weights_path = f"results/weights/{machine_id}/encoder.weights.h5"
        
        try:
            trainer.encoder.load_weights(weights_path)
            return trainer, p_dim, r_dim
        except ValueError as e:
            msg = str(e)
            match = re.search(r"Received: value\.shape=\((\d+),", msg)
            if match:
                actual_res = int(match.group(1))
                actual_phy = total_dim - actual_res
                print(f"🔄 {machine_id} Topology Correction: {actual_phy}/{actual_res}")
                return build_and_load(actual_phy, actual_res)
            else: raise e

    trainer, final_p, final_r = build_and_load(p_guess, r_guess)
    data_dict = {
        'phy': np.expand_dims(X_raw[:, :, :final_p], axis=1),
        'res': X_raw[:, :, final_p:],
        'labels': L_raw
    }
    return trainer, data_dict

# --- MAIN GLOBAL PROJECTION SCRIPT ---

def run_global_manifold(machine_list, data_root, config):
    global_z_sys, global_z_res = [], []
    global_labels, machine_tags = [], []

    print(f"🌐 Memory-Safe Aggregation of {len(machine_list)} machines...")

    for mid in machine_list:
        trainer, data_dict = get_trained_trainer_for_machine(mid, data_root, config)
        
        total_windows = len(data_dict['phy'])
        idx = np.random.choice(total_windows, min(1000, total_windows), replace=False)
        
        mini_batch_size = 256
        z_sys_machine, z_res_machine = [], []
        
        for j in range(0, len(idx), mini_batch_size):
            batch_idx = idx[j : j + mini_batch_size]
            p_mini = tf.cast(data_dict['phy'][batch_idx, 0, :, :], tf.float32)
            r_mini = tf.cast(data_dict['res'][batch_idx], tf.float32)
            
            zs, zr, _ = trainer.encoder([p_mini, r_mini], training=False)
            z_sys_machine.append(zs.numpy())
            z_res_machine.append(zr.numpy())
        
        global_z_sys.append(np.concatenate(z_sys_machine, axis=0))
        global_z_res.append(np.concatenate(z_res_machine, axis=0))
        global_labels.append(np.any(data_dict['labels'][idx] > 0, axis=1))
        machine_tags.extend([mid] * len(idx))
        
        tf.keras.backend.clear_session()
        print(f"✅ Finished {mid}")

    # --- THE MISSING LOGIC: T-SNE & PLOTTING ---
    Z_S = np.concatenate(global_z_sys, axis=0)
    Z_R = np.concatenate(global_z_res, axis=0)
    LBL = np.concatenate(global_labels, axis=0)
    TAGS = np.array(machine_tags)

    print("🪄 Starting t-SNE math (this may take 2-5 minutes)...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_jobs=-1)
    
    print("🪄 Computing HNN Consensus Space...")
    z_s_2d = tsne.fit_transform(Z_S)
    
    print("🪄 Computing TCN Residual Space...")
    z_r_2d = tsne.fit_transform(Z_R)

    print("🖼️ Rendering Plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot HNN Global
    sns.scatterplot(x=z_s_2d[~LBL, 0], y=z_s_2d[~LBL, 1], hue=TAGS[~LBL], 
                    ax=ax1, palette="tab20", alpha=0.4, s=10, legend='full')
    ax1.scatter(z_s_2d[LBL, 0], z_s_2d[LBL, 1], c='red', marker='x', s=20, label="Anomaly", alpha=0.7)
    ax1.set_title("Global HNN Consensus Manifold (Physics)")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)

    # Plot TCN Global
    sns.scatterplot(x=z_r_2d[~LBL, 0], y=z_r_2d[~LBL, 1], hue=TAGS[~LBL], 
                    ax=ax2, palette="tab20", alpha=0.4, s=10, legend=False)
    ax2.scatter(z_r_2d[LBL, 0], z_r_2d[LBL, 1], c='red', marker='x', s=20, alpha=0.7)
    ax2.set_title("Global TCN Residual Manifold (Lone Wolves)")

    os.makedirs("results/plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/plots/global_manifold_comparison.png")
    print("📁 Plot saved to results/plots/global_manifold_comparison.png")
    plt.show()

if __name__ == "__main__":
    machines = [
        "machine-1-1", "machine-1-2", "machine-1-3", "machine-1-4", "machine-1-5", "machine-1-6", "machine-1-7", "machine-1-8",
        "machine-2-1", "machine-2-2", "machine-2-3", "machine-2-4", "machine-2-5", "machine-2-6", "machine-2-7", "machine-2-8", "machine-2-9",
        "machine-3-1", "machine-3-2", "machine-3-3", "machine-3-4", "machine-3-5", "machine-3-6", "machine-3-7", "machine-3-8", "machine-3-9", "machine-3-10", "machine-3-11"
    ]
    conf = {
        "window_size": 64, "latent_dim": 512, "hnn_feature_dim": 256,
        "hnn_steps": 3, "hnn_dt": 0.1, "lr": 1e-4, "patience": 5, 
        "batch_size": 128, "dropout": 0.1, "recon_weight": 10.0,
        "lambda_d": 1.0, "lambda_e": 1.0, "alpha_vpo": 2.0, "sentinel_weight": 5.0
    }
    data_path = "/home/utsab/Downloads/smd/ServerMachineDataset"
    run_global_manifold(machines, data_path, conf)