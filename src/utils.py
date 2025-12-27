import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler

# Internal imports
from .masking import get_masked_views, resi_masker
from .router import route_features

def load_txt_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return np.loadtxt(path, delimiter=',', dtype=np.float32)
    except:
        return np.loadtxt(path, dtype=np.float32)

def calculate_physics_jerk(data, savgol_len, savgol_poly):
    """Triple-gradient Jerk calculation[cite: 7, 168]."""
    smooth = savgol_filter(data, window_length=savgol_len, polyorder=savgol_poly, axis=0)
    first = np.gradient(smooth, axis=0)
    acc = np.gradient(first, axis=0)
    jerk = np.gradient(acc, axis=0)
    return np.abs(jerk)

def create_spline_envelopes(data, window_size, sparsity_factor):
    """Spline-based boundary tracking."""
    series = pd.Series(data)
    n = len(data)
    raw_upper = series.rolling(window=window_size, center=True).max().bfill().ffill().values
    raw_lower = series.rolling(window=window_size, center=True).min().bfill().ffill().values
    
    indices = np.arange(0, n, sparsity_factor)
    if indices[-1] != n - 1:
        indices = np.append(indices, n - 1)
        
    cs_upper = CubicSpline(indices, raw_upper[indices], bc_type='natural')
    cs_lower = CubicSpline(indices, raw_lower[indices], bc_type='natural')
    
    x_new = np.arange(n)
    return cs_upper(x_new), cs_lower(x_new)

def load_smd_windows(data_root, machine_id, config):
    window = config['window_size'] # [cite: 544, 580]
    stride = config['stride'] # [cite: 544, 577]
    savgol_len = config['savgol_len']
    savgol_poly = config['savgol_poly']
    sparsity = config['sparsity_factor']

    print(f"🚀 Dual-Anchor Pipeline Initiated: {machine_id}")
    
    # 1. Loading & Scaling [cite: 234, 273]
    train_raw = load_txt_file(os.path.join(data_root, "train", f"{machine_id}.txt"))
    test_raw = load_txt_file(os.path.join(data_root, "test", f"{machine_id}.txt"))
    test_labels = load_txt_file(os.path.join(data_root, "test_label", f"{machine_id}.txt")).flatten().astype(np.int32)

    scaler = StandardScaler()
    train_total_norm = scaler.fit_transform(train_raw)
    test_total_norm = scaler.transform(test_raw)

    # 2. Physical Routing (Deterministic Agglo)
    (train_phy, train_res, test_phy, test_res), topo, phy_labels = route_features(
        train_total_norm, test_total_norm
    )
    
    # 3. Envelope Generation for Lone Wolves
    res_envelopes_upper = np.zeros_like(train_res)
    res_envelopes_lower = np.zeros_like(train_res)

    for local_idx in topo.res_to_lone_local:
        up, lo = create_spline_envelopes(train_res[:, local_idx], window, sparsity)
        res_envelopes_upper[:, local_idx] = up
        res_envelopes_lower[:, local_idx] = lo

    # 4. Triple Jerk of Noise Floor
    jerk_upper = calculate_physics_jerk(res_envelopes_upper, savgol_len, savgol_poly)
    jerk_lower = calculate_physics_jerk(res_envelopes_lower, savgol_len, savgol_poly)
    total_res_jerk = (jerk_upper + jerk_lower) / 2.0
    
    # 5. Windowing [cite: 234, 275]
    def create_windows(data, current_stride):
        num_windows = (data.shape[0] - window) // current_stride + 1 
        return np.array([data[i*current_stride : i*current_stride + window] for i in range(num_windows)], dtype=np.float32)

    train_w_phy = create_windows(train_phy, stride)
    train_res_w = create_windows(train_res, stride)
    train_res_jerk_w = create_windows(total_res_jerk, stride)
    train_jerk_phy_w = create_windows(calculate_physics_jerk(train_phy, savgol_len, savgol_poly), stride)

    # 6. Masking Logic [cite: 26, 300, 327]
    v1, v2, v3, v4 = get_masked_views(train_w_phy, train_jerk_phy_w, phy_labels)
    # Consensus: (N, 6, 64, F_phy)
    phy_views = np.stack([train_w_phy, v1, v2, v3, v4, train_jerk_phy_w], axis=1)
    # Residual: (N, 64, F_res)
    rv1 = resi_masker(train_res_w, train_res_jerk_w, p_tile=config['p_tile'])

    train_final = {
        "phy_views": phy_views,
        "res_views": rv1,
        "topology": topo
    }

    test_final = {
        "phy": create_windows(test_phy, window), 
        "res": create_windows(test_res, window),
        "topology": topo
    }

    # 7. ACAE Label Slicing [cite: 485, 486]
    test_labels = test_labels[:test_final["phy"].shape[0] * window] 

    return train_final, test_final, test_labels, scaler.mean_, scaler.scale_

def build_tf_datasets(train_final, test_final, val_split=0.2, batch_size=128):
    """Synchronized TF pipeline for Dual-Anchor branches."""
    # FIXED: Key was 'res_windows', now 'res_views' to match load_smd_windows
    phy_data = train_final['phy_views']   
    res_data = train_final['res_views'] 
    num_samples = len(phy_data)

    num_train = int(num_samples * (1 - val_split))
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]
    
    train_phy, train_res = phy_data[train_idx], res_data[train_idx]
    val_phy, val_res = phy_data[val_idx], res_data[val_idx]
    
    test_phy, test_res = test_final['phy'], test_final['res']

    AUTOTUNE = tf.data.AUTOTUNE

    def make_dataset(phy, res, shuffle=False):
        ds_phy = tf.data.Dataset.from_tensor_slices(phy)
        ds_res = tf.data.Dataset.from_tensor_slices(res)
        ds = tf.data.Dataset.zip((ds_phy, ds_res))
        if shuffle:
            ds = ds.cache().shuffle(2048)
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    return (make_dataset(train_phy, train_res, shuffle=True), 
            make_dataset(val_phy, val_res), 
            make_dataset(test_phy, test_res))