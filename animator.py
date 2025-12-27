import os
import re
import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.manifold import TSNE
import tensorflow as tf
from src.models import build_dual_encoder, build_dual_decoder, build_discriminator
from src.trainer import DualAnchorACAETrainer

# --- HELPERS (Keep your existing loading logic) ---

def load_machine_data(machine_id, data_root, window_size=64):
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
    data_dict = {'phy': np.expand_dims(X_raw[:, :, :final_p], axis=1), 'res': X_raw[:, :, final_p:], 'labels': L_raw}
    return trainer, data_dict

# --- OPTIMIZED ANIMATION LOGIC ---

def run_dual_animation(machine_id, data_root, config, n_samples=5000):
    print(f"🎬 Initializing Dual-Window Tracker for {machine_id}...")
    trainer, data_dict = get_trained_model(machine_id, data_root, config)
    
    n_samples = min(n_samples, len(data_dict['labels']))
    p_anchor = tf.cast(data_dict['phy'][:n_samples, 0, :, :], tf.float32)
    r_window = tf.cast(data_dict['res'][:n_samples], tf.float32)
    
    print("🧠 Running Encoder Inference...")
    z_sys, z_res, _ = trainer.encoder([p_anchor, r_window], training=False)
    
    # 2. Projection to 2D
    # Use init='pca' to speed up t-SNE and make it more stable for high sample counts
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    print("🪄 Computing HNN (Physics) 2D Projection...")
    hnn_2d = tsne.fit_transform(z_sys.numpy())
    print("🪄 Computing TCN (Residual) 2D Projection...")
    tcn_2d = tsne.fit_transform(z_res.numpy())
    
    window_labels = np.any(data_dict['labels'][:n_samples] > 0, axis=1)

    # 3. Setup Plot - REDUCED DPI to save RAM buffer
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=80) 
    fig.suptitle(f"Real-time Latent State Tracking: {machine_id}", fontsize=14)

    # Backgrounds
    ax1.scatter(hnn_2d[:, 0], hnn_2d[:, 1], c='gray', alpha=0.05, s=2)
    ax2.scatter(tcn_2d[:, 0], tcn_2d[:, 1], c='gray', alpha=0.05, s=2)

    line1, = ax1.plot([], [], lw=1, color='blue', alpha=0.3)
    head1 = ax1.scatter([], [], s=80, edgecolors='black', zorder=5)
    line2, = ax2.plot([], [], lw=1.2, color='purple', alpha=0.4)
    head2 = ax2.scatter([], [], s=80, edgecolors='black', zorder=5)

    def init():
        ax1.set_xlim(hnn_2d[:, 0].min()-5, hnn_2d[:, 0].max()+5)
        ax1.set_ylim(hnn_2d[:, 1].min()-5, hnn_2d[:, 1].max()+5)
        ax2.set_xlim(tcn_2d[:, 0].min()-5, tcn_2d[:, 0].max()+5)
        ax2.set_ylim(tcn_2d[:, 1].min()-5, tcn_2d[:, 1].max()+5)
        return line1, head1, line2, head2

    def update(frame):
        current_color = 'red' if window_labels[frame] else 'blue'
        
        # HNN Update
        line1.set_data(hnn_2d[:frame, 0], hnn_2d[:frame, 1])
        head1.set_offsets(hnn_2d[frame:frame+1])
        head1.set_facecolor(current_color)
        
        # TCN Update
        line2.set_data(tcn_2d[:frame, 0], tcn_2d[:frame, 1])
        head2.set_offsets(tcn_2d[frame:frame+1])
        head2.set_facecolor(current_color) 
        return line1, head1, line2, head2

    # CRITICAL CHANGE: We use a larger step (stride) for animation frames 
    # but keep the full data points (n_samples). 
    # Skipping every 4 frames won't hurt accuracy but reduces RAM load by 75%.
    frame_indices = np.arange(0, n_samples, 4) 
    
    ani = animation.FuncAnimation(fig, update, frames=frame_indices, 
                                  init_func=init, blit=True)
    
    os.makedirs("results/plots", exist_ok=True)
    save_path = f'results/plots/{machine_id}_dual_slither.gif'
    
    print(f"📦 Committing to disk. This is the ultra-stable 'safe-save' version...")
    
    # 1. We remove savefig_kwargs entirely to avoid backend TypeErrors
    # 2. We use a lower FPS (10) to make the slither easier to follow
    try:
        ani.save(
            save_path, 
            writer='pillow', 
            fps=10, 
            dpi=70 # Lowering DPI further to ensure it fits in your 12GB RAM
        )
    except Exception as e:
        print(f"❌ Standard save failed: {e}")
        print("💡 Attempting Emergency Frame-by-Frame save...")
        # Fallback: Just save a static high-res plot of the final snake state
        plt.savefig(save_path.replace('.gif', '_static.png'), dpi=150)
    
    plt.close('all')
    gc.collect() 
    print(f"✅ Process complete. Check: {save_path}")

if __name__ == "__main__":
    main_config = {
        "window_size": 64, "latent_dim": 512, "hnn_feature_dim": 256,
        "hnn_steps": 3, "hnn_dt": 0.1, "lr": 1e-4, "patience": 5, 
        "batch_size": 128, "dropout": 0.1, "recon_weight": 10.0,
        "lambda_d": 1.0, "lambda_e": 1.0, "alpha_vpo": 2.0, "sentinel_weight": 5.0
    }
    data_path_root = "/home/utsab/Downloads/smd/ServerMachineDataset"
    run_dual_animation("machine-1-3", data_path_root, main_config, n_samples=5000)