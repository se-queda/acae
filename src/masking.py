import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# --- LATENT MIXING ---

def mix_features(z_orig, z_other):
    """
    Interpolates between anchor and augmented latents for the discriminator task.
    """
    batch_size = tf.shape(z_orig)[0]
    alpha = tf.random.uniform((batch_size, 1), minval=0.0, maxval=1.0)
    z_mixed = alpha * z_orig + (1.0 - alpha) * z_other
    return z_mixed, alpha

# --- UNIVARIATE PHYSICS MASKING ---

def univariate_masker(data_windows, jerk_windows):
    N, T, F = data_windows.shape
    tiers = [95, 90, 85, 80]
    views = []

    # budgets[tier_idx, feature_idx] stores K for each tier/feature
    budgets = np.zeros((len(tiers), F), dtype=int)
    # sorted_indices[feature_idx] stores window indices sorted by jerk mean (ascending)
    sorted_indices = np.zeros((F, N), dtype=int)

    # --- Step 1: Pre-calculate Physics Stats per Feature ---
    for f in range(F):
        feat_jerk_wins = jerk_windows[:, :, f] # Shape: (N, T)
        flat_jerk = feat_jerk_wins.flatten()
        
        # Calculate all 4 thresholds in one pass
        thresholds = np.percentile(flat_jerk, tiers)
        
        for i, threshold in enumerate(thresholds):
            count_above = np.sum(flat_jerk > threshold)
            k = int(np.ceil(count_above / T))
            # Safety: At least 1 window if any instability exists
            if count_above > 0 and k == 0: k = 1
            budgets[i, f] = k
            
        # Score every window once and sort them
        window_means = np.mean(feat_jerk_wins, axis=1) # Shape: (N,)
        sorted_indices[f] = np.argsort(window_means)

    # --- Step 2: Build the 4 Tiered Views ---
    for i, p in enumerate(tiers):
        v_tier = data_windows.copy() # Template for this tier
        for f in range(F):
            k = budgets[i, f]
            if k > 0:
                # Mask the top K 'jerkiest' windows using pre-sorted indices
                # No Veto check here: if it's high jerk, it gets masked.
                top_k_idx = sorted_indices[f][-k:]
                v_tier[top_k_idx, :, f] = 0.0
        views.append(v_tier)

    return views[0], views[1], views[2], views[3]


def consensus_masker(data_windows, jerk_windows, cluster_labels):
    """
    The Final Algorithm:
    1. Runs the univariate masker for all 4 percentiles.
    2. For each view and each window, checks for cluster-wide agreement.
    3. If a masked feature's cluster isn't 100% masked, the whole cluster is unmasked.
    """
    # Step 1: Get the 4 percentile views from the univariate masker
    base_views = univariate_masker(data_windows, jerk_windows)
    
    N, T, F = data_windows.shape
    final_views = []

    for v_tier in base_views:
        # Pre-identify which (window, feature) pairs are masked (where data is zeroed)
        # Shape: (N, F)
        is_masked = np.all(v_tier == 0.0, axis=1)
        
        # Working copy for the consensus adjustments
        v_consensus = v_tier.copy()

        for n in range(N):
            processed_clusters = set() # Track processed clusters for this window
            
            for f in range(F):
                c_id = cluster_labels[f]
                
                # Skip if we already validated or vetoed this cluster in this window
                if c_id in processed_clusters:
                    continue
                
                # Check if this feature is masked
                if is_masked[n, f]:
                    # Find all features belonging to this cluster
                    c_indices = np.where(cluster_labels == c_id)[0]
                    
                    # CONSENSUS CHECK: Are ALL features in this cluster masked?
                    if np.all(is_masked[n, c_indices]):
                        # Cluster consensus reached: keep the mask
                        pass
                    else:
                        # VETO: Cluster does not agree. Unmask everyone in the group.
                        v_consensus[n, :, c_indices] = data_windows[n, :, c_indices]
                    
                    # Mark this cluster as done for this window
                    processed_clusters.add(c_id)
        
        final_views.append(v_consensus)

    return tuple(final_views)

# Point the main entry point to the new consensus logic
get_masked_views = consensus_masker
