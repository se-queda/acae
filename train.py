import yaml
import argparse
import os
import csv
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from tensorflow.keras import backend as K

from src.trainer import ACAETrainer
# Import the new projection head builder
from src.models import build_encoder, build_decoder, build_discriminator, build_projection_head
from src.utils import load_smd_windows, build_tf_datasets

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_best_f1(scores, labels, step_size=100):
    """
    Finds the best F1 score by sweeping through thresholds.
    Standard practice in Anomaly Detection papers (referenced in Sec 3.7).
    """
    min_score, max_score = np.min(scores), np.max(scores)
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_thresh = 0.0

    # Optimize threshold search (vectorized or coarse-to-fine)
    thresholds = np.linspace(min_score, max_score, step_size)
    
    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_thresh = thresh
            
    return best_f1, best_precision, best_recall, best_thresh

def train_on_machine(machine_id, config):
    print(f"\n🚀 Training on {machine_id}...")
    
    # 1. Load Data (Point-wise Evaluation Mode)
    # Note: Using stride=window (64) for testing to get non-overlapping segments
    train_w, test_w, test_labels_pointwise, _, _ = load_smd_windows(
        data_root="data/smd",
        machine_id=machine_id,
        window=64,
        train_stride=1 # Overlapping for training
    )

    train_ds, val_ds, test_ds = build_tf_datasets(
        train_w, test_w,
        val_split=0.2,
        batch_size=config['batch_size']
    )

    # 2. Build Models
    features = train_w.shape[-1] # 38 for SMD
    
    # Projection Head: (64, 38) -> (32, 128)
    projection_head = build_projection_head(input_shape=(64, features))
    
    # Encoder: Takes (32, 128) -> Latent
    encoder = build_encoder(input_shape_projected=(32, 128), latent_dim=config['latent_dim'])
    
    # Decoder: Latent -> (64, 38)
    decoder = build_decoder(latent_dim=config['latent_dim'], output_steps=64, output_features=features)
    
    # Discriminator: Latent*2 -> 2
    discriminator = build_discriminator(latent_dim=config['latent_dim'])

    # 3. Trainer
    trainer = ACAETrainer(projection_head, encoder, decoder, discriminator, config)
    trainer.fit(train_ds, val_ds=val_ds, epochs=config['epochs'])

    # 4. Evaluation (Point-wise)
    # Get window errors: (Num_Windows,)
    # For non-overlapping windows, we need to map this to point-wise errors.
    # Method A: Assign the window's MSE to every point in that window.
    # Method B: Reconstruct and calculate point-wise MSE (Better/More Exact).
    
    # Let's do Method B (Exact Point-wise Reconstruction Error)
    _, reconstructions = trainer.reconstruct(test_ds)
    # reconstructions shape: (N_windows, 64, 38)
    # Flatten to: (N_points, 38)
    recons_flat = reconstructions.reshape(-1, features)
    
    # Get original test data (flattened)
    # We can reconstruct it from test_w or just reload/reshape
    originals_flat = test_w.reshape(-1, features)
    
    # Calculate Point-wise Anomaly Score (Sum of Squared Errors per timestamp)
    # Shape: (N_points,)
    point_scores = np.sum(np.square(originals_flat - recons_flat), axis=1)
    
    # Ensure lengths match (load_smd_windows handles the label trimming)
    min_len = min(len(point_scores), len(test_labels_pointwise))
    point_scores = point_scores[:min_len]
    test_labels_pointwise = test_labels_pointwise[:min_len]

    # Calculate AUC (Point-wise)
    auc = roc_auc_score(test_labels_pointwise, point_scores)
    
    # Calculate Best F1 (Point-wise)
    best_f1, prec, rec, thresh = get_best_f1(point_scores, test_labels_pointwise)
    
    print(f"✅ {machine_id} -> AUC: {auc:.4f} | Best F1: {best_f1:.4f} (P={prec:.4f}, R={rec:.4f})")

    # Save Checkpoints
    machine_dir = f"checkpoints/{machine_id}"
    os.makedirs(machine_dir, exist_ok=True)
    projection_head.save_weights(f"{machine_dir}/projection.weights.h5")
    encoder.save_weights(f"{machine_dir}/encoder.weights.h5")
    decoder.save_weights(f"{machine_dir}/decoder.weights.h5")
    discriminator.save_weights(f"{machine_dir}/discriminator.weights.h5")

    K.clear_session()

    return auc, best_f1

def run_all_machines(config):
    machine_ids = (
            [f"machine-1-{i}" for i in range(1, 9)] +
            [f"machine-2-{i}" for i in range(1, 10)] +
            [f"machine-3-{i}" for i in range(1, 12)]
    )
    # Adjust skip list as needed
    skip_machines = set([]) 
    
    results = []
    for mid in machine_ids:
        if mid in skip_machines:
            continue
        try:
            auc, f1 = train_on_machine(mid, config)
            results.append((mid, auc, f1))
        except Exception as e:
            print(f"❌ Failed on {mid}: {e}")

    print("\n📊 Final Scores Across All Machines:")
    print(f"{'Machine':15s} | {'AUC':10s} | {'F1':10s}")
    for mid, auc, f1 in results:
        print(f"{mid:15s} | {auc:.4f}     | {f1:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/smd_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["machine_id", "auc", "best_f1"])
        writer.writerows(results)

    print("\n💾 Results saved to: results/smd_results.csv")

def main(config_path, run_all=False):
    config = load_config(config_path)

    if run_all:
        run_all_machines(config)
    else:
        train_on_machine("machine-1-1", config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ACAE on smd")
    parser.add_argument('--config', type=str, default="config.yaml", help='Path to config file')
    parser.add_argument('--all', action='store_true', help='Train on all smd machines')
    args = parser.parse_args()

    main(args.config, run_all=args.all)
