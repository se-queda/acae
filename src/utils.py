import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os
from scipy.signal import savgol_filter
from .masking import get_masked_views
from sklearn.cluster import AgglomerativeClustering

def load_txt_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return np.loadtxt(path, delimiter=',', dtype=np.float32)
    except:
        return np.loadtxt(path, dtype=np.float32)


def preprocess_and_cluster(train_data, dist_threshold=0.5):
    """
    1. Removes constant (dead) features.
    2. Clusters remaining features via correlation.
    3. Purges lone-wolf features (clusters of size 1).
    Returns cleaned training data, cluster labels, and surviving feature indices.
    """
    # 1. Identify active (non-constant) features
    variances = np.var(train_data, axis=0)
    active_feat_indices = np.where(variances > 1e-9)[0]
    train_active = train_data[:, active_feat_indices]

    # 2. Correlation-based grouping
    corr = np.nan_to_num(np.corrcoef(train_active, rowvar=False))
    dist = 1 - np.abs(corr)
    
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_threshold, linkage='complete')
    labels = clustering.fit_predict(dist)

    # 3. Identify consensus-capable clusters (size > 1)
    unique_ids, counts = np.unique(labels, return_counts=True)
    multi_feature_ids = unique_ids[counts > 1]
    
    # Feature selector determines which COLUMNS to keep
    feature_selector = np.isin(labels, multi_feature_ids)
    
    final_train_data = train_active[:, feature_selector]
    final_cluster_labels = labels[feature_selector]
    final_feature_indices = active_feat_indices[feature_selector]

    return final_train_data, final_cluster_labels, final_feature_indices
def calculate_physics_jerk(data, savgol_len=11, savgol_poly=3):
    """
    The Physics Branch: Converts raw telemetry into a 'Jerk Matrix'.
    Processes all 38 features in parallel using vectorized gradients.
    """
    # 1. Smooth the entire sequence (axis=0) to remove sensor noise
    smooth = savgol_filter(data, window_length=savgol_len, polyorder=savgol_poly, axis=0)
    
    # 2. Triple-gradient to find the Jerk (3rd derivative)
    first = np.gradient(smooth, axis=0)
    acc = np.gradient(first, axis=0)
    jerk = np.gradient(acc, axis=0)
    
    return np.abs(jerk)

def load_smd_windows(data_root, machine_id, window=64, train_stride=1):
    """
    Updated Load Module: Creates 5 parallel window sets.
    1. Main Normed Data
    2. Jerk View 1 (T95)
    3. Jerk View 2 (T90)
    4. Jerk View 3 (T85)
    5. Jerk View 4 (T80)
    """
    # --- Step 1: Loading & Scaling ---
    print(f"🚀 Processing {machine_id} with Global Jerk strategy...")
    train_path = os.path.join(data_root, "train", f"{machine_id}.txt")
    test_path = os.path.join(data_root, "test", f"{machine_id}.txt")
    label_path = os.path.join(data_root, "test_label", f"{machine_id}.txt")

    train_raw = load_txt_file(train_path)
    test_raw = load_txt_file(test_path)
    test_labels = load_txt_file(label_path).flatten()

    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_raw)
    test_norm = scaler.transform(test_raw)

    # Removes constant features and groups others by correlation
    train_norm, clusters, live_feat_idx= preprocess_and_cluster(train_norm)
    test_norm_filtered = test_norm[:, live_feat_idx]

    
    # --- Step 2: Global Branching ---
    # We calculate the Jerk Matrix for the WHOLE sequence once.
    train_jerk_matrix = calculate_physics_jerk(train_norm)

    def create_windows(data, stride):
        num_windows = (data.shape[0] - window) // stride + 1
        return np.array([
            data[i*stride : i*stride + window]
            for i in range(num_windows)
        ], dtype=np.float32)

    # Branch A: Main normalized windows
    train_w_main = create_windows(train_norm, stride=train_stride)
    # Branch B: Jerk windows for the masker
    train_jerk_w = create_windows(train_jerk_matrix, stride=train_stride)

    # --- Step 3: Call Masker Module ---
    # We'll define masker.py later, but we call it here to get our 4 tiers.
    # It takes the jerk windows and outputs 4 binary masks.
    v1, v2, v3, v4 = get_masked_views(train_w_main, train_jerk_w, clusters)

    # Pack the 5 parallel views together: (N, 5, window, features)
    train_packed = np.stack([train_w_main, v1, v2, v3, v4, train_jerk_w], axis=1)
    
    # Test remains standard (no masking needed for evaluation)
    test_w = create_windows(test_norm_filtered, stride=window)
    test_labels = test_labels[:test_w.shape[0] * window]

    return train_packed, test_w, test_labels, scaler.mean_[live_feat_idx], scaler.scale_[live_feat_idx]

def build_tf_datasets(train_w, test_w, val_split=0.2, batch_size=128):
    """
    Wraps the 5 parallel views into an optimized TF pipeline.
    """
    # Split Train into Train/Val
    num_train = int(len(train_w) * (1 - val_split))
    
    # Shuffle indices for random split
    indices = np.arange(len(train_w))
    np.random.shuffle(indices)
    
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]
    
    train_data = train_w[train_idx]
    val_data = train_w[val_idx]

    AUTOTUNE = tf.data.AUTOTUNE

    def make_dataset(data, shuffle=False):
        # Slices (N, 5, 64, 38) -> yields (5, 64, 38)
        ds = tf.data.Dataset.from_tensor_slices(data)
        if shuffle:
            ds = ds.cache().shuffle(2048)
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    train_ds = make_dataset(train_data, shuffle=True)
    val_ds = make_dataset(val_data, shuffle=False)
    test_ds = make_dataset(test_w, shuffle=False)

    return train_ds, val_ds, test_ds
