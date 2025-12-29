import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
from src.models import build_dual_encoder, build_dual_decoder, build_discriminator
from src.trainer import DualAnchorACAETrainer
import seaborn as sns
import re
import pandas as pd
from src.utils import load_txt_file

def load_psm_data_for_viz(data_root, window_size=64):
    """
    Loads PSM CSV data, handles missing values, and prepares windowed samples.
    """
    print("📂 Loading PSM data for visualization...")
    # Load and clean telemetry
    test_df = pd.read_csv(os.path.join(data_root, "test.csv")).ffill().bfill()
    label_df = pd.read_csv(os.path.join(data_root, "test_label.csv"))
    
    # PSM has a timestamp in col 0; slice it out
    test_data = test_df.values[:, 1:].astype(float)
    labels = label_df.values[:, 1:].astype(float)
    
    total_dim = test_data.shape[-1]
    
    # Matching your stable PSM topology: 13 Consensus / 12 Residual
    n_phy = 13 
    n_res = total_dim - n_phy
    
    X, L = [], []
    for i in range(len(test_data) - window_size + 1):
        X.append(test_data[i : i + window_size])
        L.append(labels[i : i + window_size])
    
    X = np.array(X)
    L = np.array(L)

    return {
        'phy': np.expand_dims(X[:, :, :n_phy], axis=1), 
        'res': X[:, :, n_phy:],
        'labels': L,
        'dims': (n_phy, n_res)
    }

def generate_sentinel_heatmap(trainer, test_final, topo, machine_name="Machine"):
    """
    Visualizes the reconstruction error of the Dead Sentinels.
    """
    print(f"🌡️ Generating Sentinel Heatmap for {machine_name}...")
    
    # 1. Reconstruct Data
    # In the test stage, ACAE reconstructs test samples using the trained weights [cite: 485, 486]
    recons = trainer.reconstruct(test_final)
    
    # Extract original and reconstructed residuals
    res_orig = recons['res_orig'] # Shape: (N, W, 14)
    res_hat = recons['res_hat']   # Shape: (N, W, 14)
    
    # 2. Isolate Dead Sentinels (5 sensors)
    # These indices were identified as stagnant during preprocessing
    dead_indices = topo.res_to_dead_local 
    sentinel_orig = res_orig[:, :, dead_indices]
    sentinel_hat = res_hat[:, :, dead_indices]
    
    # 3. Calculate Point-wise MSE for Sentinels
    # Anomaly scores are defined as squared reconstruction errors [cite: 486, 487]
    sentinel_error = np.square(sentinel_orig - sentinel_hat)
    
    # Aggregate over the window for visualization (Mean over W)
    # Shape becomes (N, 5)
    heatmap_data = np.mean(sentinel_error, axis=1).T 

    # 4. Plotting
    plt.figure(figsize=(15, 6))
    sns.heatmap(heatmap_data, cmap="YlOrRd", cbar_kws={'label': 'Reconstruction Error'})
    
    # Overlay Ground Truth Anomaly Regions
    window_labels = np.any(test_final['labels'] > 0, axis=1)
    anomaly_indices = np.where(window_labels)[0]
    
    # Draw vertical lines for anomalies
    for idx in anomaly_indices:
        plt.axvline(x=idx, color='red', alpha=0.05, linewidth=1)

    plt.title(f"Dead Sentinel Error Heatmap: {machine_name}\n(Red Vertical Bars = Ground Truth Anomalies)")
    plt.xlabel("Time Window Index")
    plt.ylabel("Sentinel Sensor Index")
    plt.tight_layout()
    plt.savefig(f"results/plots/{machine_name}_sentinel_heatmap.png")
    plt.show()

def visualize_dual_latent_space(config, test_final, weights_path, machine_name="PSM"):
    print(f"🚀 Initializing Latent Analysis for {machine_name}...")
    
    W = config.get("window_size", 64)
    phy_dim, res_dim = test_final['dims']
    
    # 1. Rebuild Encoder
    encoder = build_dual_encoder((W, phy_dim), (W, res_dim), config)
    
    # 2. Load Weights
    weights_file = os.path.join(weights_path, "encoder.weights.h5")
    if os.path.exists(weights_file):
        encoder.load_weights(weights_file)
        print(f"✅ Loaded weights from {weights_file}")
    else:
        print(f"⚠️ Warning: Weights not found at {weights_file}")

    # 3. Extract Data & Handle Label Dimensions
    phy_anchor = tf.cast(test_final['phy'][:, 0, :, :], tf.float32)
    res_windows = tf.cast(test_final['res'], tf.float32)
    
    # FIXED: Use axis=(1, 2) to flatten the (N, W, 1) or (N, W) labels 
    # into a clean (N,) boolean vector 
    window_labels = np.any(test_final['labels'] > 0, axis=(1, 2)) if test_final['labels'].ndim == 3 \
                    else np.any(test_final['labels'] > 0, axis=1)
    
    # Ensure it is strictly 1D to prevent IndexError during scatter plotting
    window_labels = window_labels.flatten()

    # --- FIXED: Batched Inference to prevent GPU OOM ---
    print(f"🪄 Extracting features for {len(phy_anchor)} windows in batches...")
    batch_size = 128
    z_sys_list, z_res_list = [], []
    
    for i in range(0, len(phy_anchor), batch_size):
        p_batch = phy_anchor[i : i + batch_size]
        r_batch = res_windows[i : i + batch_size]
        
        # Get latent variables from encoder [cite: 381, 383]
        zs, zr, _ = encoder([p_batch, r_batch], training=False)
        z_sys_list.append(zs.numpy())
        z_res_list.append(zr.numpy())
    
    z_sys = np.concatenate(z_sys_list, axis=0)
    z_res = np.concatenate(z_res_list, axis=0)
    # -----------------------------------------------

    # 4. t-SNE Projection (Sampling for speed and clean visualization) [cite: 615]
    print("🪄 Computing t-SNE projection...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    
    # Sample subset to prevent overcrowded plots and speed up computation [cite: 615]
    num_samples = min(5000, len(z_sys))
    indices = np.random.choice(len(z_sys), num_samples, replace=False)
    
    z_s_2d = tsne.fit_transform(z_sys[indices])
    z_r_2d = tsne.fit_transform(z_res[indices])
    
    # Slice labels to match the sampled indices
    lbls = window_labels[indices]

    # 5. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Consensus Path (HNN) - Expected to be circular for PSM [cite: 624, 626]
    ax1.scatter(z_s_2d[~lbls, 0], z_s_2d[~lbls, 1], c='royalblue', label='Normal', alpha=0.4, s=8)
    ax1.scatter(z_s_2d[lbls, 0], z_s_2d[lbls, 1], c='crimson', label='Anomaly', marker='x', s=12)
    ax1.set_title(f"PSM Consensus Manifold (HNN)\nTarget: Circular Distribution [cite: 624]")
    ax1.legend()

    # Residual Path (TCN) - High-level semantic separation [cite: 633]
    ax2.scatter(z_r_2d[~lbls, 0], z_r_2d[~lbls, 1], c='seagreen', label='Normal', alpha=0.4, s=8)
    ax2.scatter(z_r_2d[lbls, 0], z_r_2d[lbls, 1], c='darkorange', label='Anomaly', marker='x', s=12)
    ax2.set_title(f"PSM Residual Manifold (TCN)\nTarget: Semantic Separation [cite: 633]")
    ax2.legend()
    
    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    save_path = f"results/plots/{machine_name}_latent_viz.png"
    plt.savefig(save_path)
    print(f"✅ Visualization saved to {save_path}")
    plt.show()
import re

def run_integrated_visualization(machine_id, data_root, config):
    print(f"🚀 Starting Integrated Analysis for {machine_id}...")
    
    # 1. Start with your current 24/14 guess
    test_final = load_raw_txt_smd_with_topology(machine_id, data_root, config["window_size"])
    phy_dim, res_dim = test_final['dims']

    def build_and_load(p_dim, r_dim):
        W, L = config["window_size"], config["latent_dim"]
        
        # Build the exact same architecture used in training
        encoder = build_dual_encoder((W, p_dim), (W, r_dim), config)
        decoder = build_dual_decoder(p_dim, r_dim, W, config)
        discriminator = build_discriminator(input_dim=L * 2)

        trainer = DualAnchorACAETrainer(
            encoder=encoder, decoder=decoder, discriminator=discriminator,
            config=config, topology=None # We'll fix topo later
        )
        
        weights_dir = f"results/weights/{machine_id}"
        try:
            trainer.encoder.load_weights(f"{weights_dir}/encoder.weights.h5")
            trainer.decoder.load_weights(f"{weights_dir}/decoder.weights.h5")
            return trainer, p_dim, r_dim
        except ValueError as e:
            # EXTRACT THE REAL SHAPE FROM ERROR STRING
            # Example: "Received: value.shape=(16, 256)"
            msg = str(e)
            match = re.search(r"Received: value\.shape=\((\d+),", msg)
            if match:
                actual_res_dim = int(match.group(1))
                actual_phy_dim = (p_dim + r_dim) - actual_res_dim
                print(f"🔄 Shape Mismatch! Rebuilding with actual dims: {actual_phy_dim}/{actual_res_dim}")
                return build_and_load(actual_phy_dim, actual_res_dim)
            else:
                raise e

    # Execute dynamic builder
    trainer, final_phy_dim, final_res_dim = build_and_load(phy_dim, res_dim)

    # 2. Correct the data split to match the weights
    total_x = np.concatenate([test_final['phy'][:, 0, :, :], test_final['res']], axis=-1)
    test_final['phy'] = np.expand_dims(total_x[:, :, :final_phy_dim], axis=1)
    test_final['res'] = total_x[:, :, final_phy_dim:]
    
    # 3. Reconstruct Topology Object
    class MachineTopology:
        def __init__(self, p_dim, r_dim):
            self.res_to_dead_local = list(range(r_dim - 5, r_dim)) # Last 5 are sentinels
    
    topo = MachineTopology(final_phy_dim, final_res_dim)
    trainer.topo = topo

    # 4. Run Plotters
    visualize_dual_latent_space(config, test_final, f"results/weights/{machine_id}", machine_id)
    generate_sentinel_heatmap(trainer, test_final, topo, machine_id)

if __name__ == "__main__":
    psm_config = {
        "window_size": 64, "latent_dim": 512, "hnn_feature_dim": 256,
        "hnn_steps": 3, "hnn_dt": 0.01, # Stable dt for PSM
        "dropout": 0.1, "lr": 1e-4
    }
    
    psm_path = "/home/utsab/Downloads/PSM"
    psm_weights = "results/weights/PSM/PSM_Pooled"
    
    # 1. Load Data
    test_data = load_psm_data_for_viz(psm_path, psm_config["window_size"])
    
    # 2. Run Visualizer
    visualize_dual_latent_space(psm_config, test_data, psm_weights, "PSM_Pooled")