import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
import numpy as np
import os
import seaborn as sns


# Assuming build_dual_encoder and load_smd_windows are already imported
from src.models import build_dual_encoder
from src.utils import load_smd_windows

def generate_latent_report(machine_id, config, test_final, test_labels):
    # 0. Setup Directories
    os.makedirs("results/latent_plots", exist_ok=True)
    
    # 1. Rebuild and Load Encoder
    # Accessing topology safely
    topo = test_final.get('topology')
    if topo is None:
        print("❌ Error: 'topology' not found in test_final. Ensure data is loaded correctly.")
        return

    phy_dim = len(topo.idx_phy)
    res_dim = len(topo.idx_res)
    
    # Build architecture
    encoder = build_dual_encoder(input_shape_sys=(64, phy_dim), input_shape_res=(64, res_dim))
    
    weight_path = f"results/weights/{machine_id}/encoder.weights.h5"
    if not os.path.exists(weight_path):
        print(f"❌ Error: Weights not found at {weight_path}")
        return

    encoder.load_weights(weight_path)
    print(f"📂 Loaded weights for {machine_id}")
    
    # 2. Inference & Data Handling
    # FIX: Use .get() to handle naming inconsistencies between 'phy'/'phy_views' and 'res'/'res_windows'
    phy_raw = test_final.get('phy') if test_final.get('phy') is not None else test_final.get('phy_views')
    res_raw = test_final.get('res') if test_final.get('res') is not None else test_final.get('res_windows')
    
    if phy_raw is None or res_raw is None:
        print(f"❌ Error: Could not find physics or residual data in test_final. Keys present: {list(test_final.keys())}")
        return

    # Slicing: Handle 4D (N, 6, 64, F) vs 3D (N, 64, F)
    if len(phy_raw.shape) == 4:
        phy_anchor = tf.cast(phy_raw[:, 0, :, :], tf.float32)
    else:
        phy_anchor = tf.cast(phy_raw, tf.float32)
        
    res_windows = tf.cast(res_raw, tf.float32)
    
    print(f"🧬 Extracting latents for {len(phy_anchor)} samples...")
    # Extract latents [z_sys, z_res, z_combined]
    z_s, z_r, _ = encoder([phy_anchor, res_windows], training=False)
    
    # 3. Dimensionality Reduction (t-SNE)
    # 5k points is the sweet spot for performance vs detail
    max_pts = min(len(z_s), 5000)
    idx = np.random.choice(len(z_s), max_pts, replace=False)
    labels_sub = test_labels[idx]
    
    print(f"🌀 Running t-SNE on {max_pts} points (this may take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    
    zs_2d = tsne.fit_transform(z_s.numpy()[idx])
    zr_2d = tsne.fit_transform(z_r.numpy()[idx])
    
# 4. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plt.suptitle(f"K-SHIELD Dual Latent Decomposition: {machine_id}", fontsize=16)
    
    # Consensus (HNN) Plot - Using 'jet' for high-vibrancy contrast
    sc1 = ax1.scatter(zs_2d[:, 0], zs_2d[:, 1], c=labels_sub, 
                     cmap='jet', s=15, alpha=1.0, edgecolors='none',
                     vmin=0, vmax=1) # Anchor colors to labels
    ax1.set_title("Consensus (HNN) Latent Space\n(System Dynamics & Physics)")
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    
    # Residual Plot - Using 'jet' to make anomalies (red) stand out
    sc2 = ax2.scatter(zr_2d[:, 0], zr_2d[:, 1], c=labels_sub, 
                     cmap='jet', s=15, alpha=1.0, edgecolors='none',
                     vmin=0, vmax=1) # Anchor colors to labels
    ax2.set_title("Residual Latent Space\n(Lone-Wolf Jitters & Sensor Noise)")
    ax2.set_xlabel("t-SNE 1")
    
    # Add a unified colorbar with fixed ticks
    cbar = fig.colorbar(sc2, ax=[ax1, ax2], location='right', aspect=30, shrink=0.8, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Normal', 'Anomaly']) 
    cbar.set_label('System State', rotation=270, labelpad=15)
    
    output_file = f"results/latent_plots/{machine_id}_vibrant_report.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🚀 Vibrant Report saved: {output_file}")





def generate_sensor_awareness_standalone(machine_id, test_final):
    """
    Rebuilds the encoder from saved weights to generate the Sensor Knowledge Map.
    """
    print(f"🔍 Analyzing sensor grouping for {machine_id} (Standalone Mode)...")
    
    # 1. Setup & Build Encoder
    topo = test_final['topology']
    phy_dim, res_dim = len(topo.idx_phy), len(topo.idx_res)
    encoder = build_dual_encoder(input_shape_sys=(64, phy_dim), input_shape_res=(64, res_dim))
    
    # Load the specific encoder weights we saved earlier
    weight_path = f"results/weights/{machine_id}/encoder.weights.h5"
    if not os.path.exists(weight_path):
        print(f"❌ Error: Weights not found at {weight_path}")
        return
    encoder.load_weights(weight_path)

    # 2. Prepare Data
    phy_raw = test_final.get('phy', test_final.get('phy_views'))
    res_raw = test_final.get('res', test_final.get('res_windows'))
    
    # Slicing for Anchor View
    phy_anchor = tf.cast(phy_raw[:, 0, :, :] if len(phy_raw.shape) == 4 else phy_raw, tf.float32)
    res_windows = tf.cast(res_raw, tf.float32)

    # 3. Gradient Attribution
    # We measure which sensors 'excite' the Residual Latent Space (z_r)
    with tf.GradientTape() as tape:
        tape.watch(res_windows)
        # encoder outputs [z_s, z_r, z_comb]
        _, z_r, _ = encoder([phy_anchor, res_windows], training=False)
        latent_magnitude = tf.reduce_sum(tf.abs(z_r))

    grads = tape.gradient(latent_magnitude, res_windows)
    # Average across time and samples for per-sensor intensity
    sensor_importance = np.mean(np.abs(grads.numpy()), axis=(0, 1))

    # 4. Plotting
    plt.figure(figsize=(16, 5))
    sns.heatmap(sensor_importance.reshape(1, -1), cmap='rocket', cbar_kws={'label': 'Residual Intensity'})
    
    # Overlay Topology Knowledge
    for idx in topo.res_to_dead_local:
        plt.axvline(x=idx + 0.5, color='#00CCFF', linestyle='--', linewidth=2, label='Dead (Topo)')
    for idx in topo.res_to_lone_local:
        plt.axvline(x=idx + 0.5, color='#00FF00', linestyle=':', linewidth=1.5, label='Lone Wolf')

    plt.title(f"K-SHIELD Sensor Knowledge Map: {machine_id}")
    plt.xlabel("Sensor Index")
    plt.yticks([])

    # Cleanup Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    output_path = f"results/latent_plots/{machine_id}_sensor_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"🔥 Standalone Awareness Heatmap saved: {output_path}")

# --- UPDATED EXECUTION ---

# 1. Load Data
_, test_final, _, _, _ = load_smd_windows(
    data_root="/home/utsab/Downloads/smd/ServerMachineDataset",
    machine_id="machine-1-1",
    window=64
)

# 2. Run the Heatmap without needing the 'trainer' variable
generate_sensor_awareness_standalone(
    machine_id="machine-1-1",
    test_final=test_final
)
# --- Execution ---
train_final, test_final, test_labels, _, _ = load_smd_windows(
    data_root="/home/utsab/Downloads/smd/ServerMachineDataset",
    machine_id="machine-1-1",
    window=64
)

generate_latent_report(
    machine_id="machine-1-1",
    config={"latent_dim": 256}, 
    test_final=test_final,
    test_labels=test_labels
)


# 2. Run the Heatmap without needing the 'trainer' variable
generate_sensor_awareness_standalone(
    machine_id="machine-1-1",
    test_final=test_final
)