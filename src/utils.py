import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os
from scipy.signal import savgol_filter
from .masking import get_masked_views
from sklearn.cluster import AgglomerativeClustering
from .router import route_features

def load_txt_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return np.loadtxt(path, delimiter=',', dtype=np.float32)
    except:
        return np.loadtxt(path, dtype=np.float32)



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
    Dual-Anchor Loader: Routes features, masks consensus groups, 
    and packs topology metadata into a structured return.
    """
    print(f"🚀 Dual-Anchor Pipeline Initiated for {machine_id}")
    
    # --- Step 1: Loading & Scaling ---
    train_path = os.path.join(data_root, "train", f"{machine_id}.txt")
    test_path = os.path.join(data_root, "test", f"{machine_id}.txt")
    label_path = os.path.join(data_root, "test_label", f"{machine_id}.txt")

    train_raw = np.loadtxt(train_path, delimiter=',', dtype=np.float32)
    test_raw = np.loadtxt(test_path, delimiter=',', dtype=np.float32)
    test_labels = np.loadtxt(label_path, delimiter=',', dtype=np.int32).flatten()

    scaler = StandardScaler()
    train_total_norm = scaler.fit_transform(train_raw)
    test_total_norm = scaler.transform(test_raw)

    # --- Step 2: Physical Routing ---
    # Now extracts the phy_labels required for the Consensus Masker Veto
    (train_phy, train_res, test_phy, test_res), topo, phy_labels = route_features(
        train_total_norm, test_total_norm
    )
    
    # --- Step 3: Physics & Windowing ---
    train_jerk_matrix = calculate_physics_jerk(train_phy)

    def create_windows(data, stride):
        num_windows = (data.shape[0] - window) // stride + 1
        return np.array([
            data[i*stride : i*stride + window]
            for i in range(num_windows)
        ], dtype=np.float32)

    train_w_phy = create_windows(train_phy, stride=train_stride)
    train_jerk_w = create_windows(train_jerk_matrix, stride=train_stride)
    train_res_w = create_windows(train_res, stride=train_stride)

    # --- Step 4: Consensus Masking ---
    # This uses the phy_labels to ensure "Cluster-Wide Agreement"
    v1, v2, v3, v4 = get_masked_views(train_w_phy, train_jerk_w, phy_labels)

    # --- Step 5: Packing Structural Returns ---
    
    # train_final: [Views(Main+4Masks+Jerk), Residual_Data, Topology]
    # Shape of phy_views: (N, 6, window, feat_phy)
    phy_views = np.stack([train_w_phy, v1, v2, v3, v4, train_jerk_w], axis=1)
    
    train_final = {
        "phy_views": phy_views,
        "res_windows": train_res_w,
        "topology": topo
    }

    # test_final: Standard windows for both anchors
    test_w_phy = create_windows(test_phy, stride=window)
    test_w_res = create_windows(test_res, stride=window)
    
    test_final = {
        "phy": test_w_phy,
        "res": test_w_res,
        "topology": topo # Also include topo in test for the custom scorer
    }

    # Align labels with test windows
    test_labels = test_labels[:test_w_phy.shape[0] * window]

    # Return signature maintained (train, test, labels, mean, scale)
    return train_final, test_final, test_labels, scaler.mean_, scaler.scale_

def build_tf_datasets(train_final, test_final, val_split=0.2, batch_size=128):
    """
    Optimized TF pipeline for Dual-Anchor dictionaries.
    Zips Phy Views and Res Windows into a single synchronized stream.
    """
    # 1. Extract Arrays from the Dictionary
    phy_data = train_final['phy_views']   # (N, 6, 64, F_phy)
    res_data = train_final['res_windows'] # (N, 64, F_res)
    num_samples = len(phy_data)

    # 2. Synchronized Train/Val Split
    num_train = int(num_samples * (1 - val_split))
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]
    
    # 3. Data Shuffling and Slicing
    train_phy, train_res = phy_data[train_idx], res_data[train_idx]
    val_phy, val_res = phy_data[val_idx], res_data[val_idx]
    
    test_phy = test_final['phy']
    test_res = test_final['res']

    AUTOTUNE = tf.data.AUTOTUNE

    def make_dataset(phy, res, shuffle=False):
        # Create separate datasets for each branch
        ds_phy = tf.data.Dataset.from_tensor_slices(phy)
        ds_res = tf.data.Dataset.from_tensor_slices(res)
        
        # Zip them so they yield (phy_batch, res_batch) together
        ds = tf.data.Dataset.zip((ds_phy, ds_res))
        
        if shuffle:
            ds = ds.cache().shuffle(2048)
        
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    # 4. Final Dataset Creation
    train_ds = make_dataset(train_phy, train_res, shuffle=True)
    val_ds = make_dataset(val_phy, val_res, shuffle=False)
    test_ds = make_dataset(test_phy, test_res, shuffle=False)

    return train_ds, val_ds, test_ds