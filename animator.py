import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from sklearn.manifold import TSNE
import seaborn as sns
from src.models import build_dual_encoder, build_dual_decoder, build_discriminator
from src.trainer import DualAnchorACAETrainer

# --- HELPER FUNCTIONS ---

def load_machine_data(machine_id, data_root, window_size=64):
    """Loads raw SMD data and labels[cite: 501, 275]."""
    test_path = os.path.join(data_root, "test", f"{machine_id}.txt")
    label_path = os.path.join(data_root, "test_label", f"{machine_id}.txt")
    
    test_data = np.loadtxt(test_path, delimiter=',')
    labels = np.loadtxt(label_path, delimiter=',')
    
    X, L = [], []
    for i in range(0, len(test_data) - window_size + 1, 1):
        X.append(test_data[i : i + window_size])
        L.append(labels[i : i + window_size])
    
    return np.array(X), np.array(L)

def get_trained_model(machine_id, data_root, config):
    """Rebuilds trainer with dynamic topology correction."""
    W = config["window_size"]
    X_raw, L_raw = load_machine_data(machine_id, data_root, W)
    total_dim = X_raw.shape[-1]

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
            match = re.search(r"Received: value\.shape=\((\d+),", str(e))
            if match:
                actual_res = int(match.group(1))
                return build_and_load(total_dim - actual_res, actual_res)
            else: raise e

    trainer, final_p, final_r = build_and_load(24, total_dim - 24)
    data_dict = {
        'phy': np.expand_dims(X_raw[:, :, :final_p], axis=1),
        'res': X_raw[:, :, final_p:],
        'labels': L_raw
    }
    return trainer, data_dict

# --- ANIMATION & EXECUTION ---

def run_trajectory_animation(machine_id, data_root, config, n_samples=3000):
    print(f"🎬 Initializing Trajectory Tracker for {machine_id}...")
    trainer, data_dict = get_trained_model(machine_id, data_root, config)
    
    # 1. Extract Latent Features
    p_anchor = tf.cast(data_dict['phy'][:n_samples, 0, :, :], tf.float32)
    r_window = tf.cast(data_dict['res'][:n_samples], tf.float32)
    
    # Encoder returns [z_sys, z_res, z_combined] [cite: 383, 423]
    _, z_res, _ = trainer.encoder([p_anchor, r_window], training=False)
    z_res = z_res.numpy()

    # 2. Projection to 2D [cite: 615]
    print("🪄 Computing t-SNE Trajectory...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_2d = tsne.fit_transform(z_res)
    
    # Anomaly labels for these points [cite: 617]
    window_labels = np.any(data_dict['labels'][:n_samples] > 0, axis=1)

    # 3. Setup Animation
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(z_2d[:, 0], z_2d[:, 1], c='gray', alpha=0.1, s=5, label="Total Manifold")
    
    line, = ax.plot([], [], lw=1.5, color='blue', alpha=0.5)
    head = ax.scatter([], [], s=60, edgecolors='black')
    
    def init():
        ax.set_xlim(z_2d[:, 0].min() - 5, z_2d[:, 0].max() + 5)
        ax.set_ylim(z_2d[:, 1].min() - 5, z_2d[:, 1].max() + 5)
        ax.set_title(f"TCN Latent 'Snake' Tracker: {machine_id}")
        return line, head

    def update(frame):
        # Update trailing path
        line.set_data(z_2d[:frame, 0], z_2d[:frame, 1])
        
        # Update current point (head)
        head.set_offsets(z_2d[frame:frame+1])
        color = 'red' if window_labels[frame] else 'blue'
        head.set_facecolor(color)
        return line, head

    ani = animation.FuncAnimation(fig, update, frames=n_samples, init_func=init, blit=True, interval=10)
    
    os.makedirs("results/plots", exist_ok=True)
# Change the extension to .gif
    save_path = f'results/plots/{machine_id}_trajectory.gif'
    print(f"💾 Saving animation to {save_path} (Pillow writer)...")
    
    # Remove the 'writer' argument to let it default to Pillow
    ani.save(save_path, fps=30) 
    plt.show()

if __name__ == "__main__":
    main_conf = {
        "window_size": 64, "latent_dim": 512, "hnn_feature_dim": 256,
        "hnn_steps": 3, "hnn_dt": 0.1, "lr": 1e-4, "patience": 5, 
        "batch_size": 128, "dropout": 0.1, "recon_weight": 10.0,
        "lambda_d": 1.0, "lambda_e": 1.0, "alpha_vpo": 2.0, "sentinel_weight": 5.0
    }
    
    data_path = "/home/utsab/Downloads/smd/ServerMachineDataset"
    # Choose a machine with a known anomaly segment for the best 'slither'
    run_trajectory_animation("machine-1-3", data_path, main_conf, n_samples=2000)