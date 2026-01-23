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


def load_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found {path}")
    try:
        return np.loadtxt(path, delimiters = ',', dtype = np.float32)
    except:
        return np.loadtxt(path, dtype = np.float32)

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