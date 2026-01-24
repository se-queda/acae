import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

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


def create_windows(data, current_stride):
    """
    Input:
        data : np.ndarray (C, T)
    Output:
        windows : np.ndarray (W, C, T)
    """
    C, T = data.shape
    num_windows = (T - window) // current_stride + 1

    return np.array(
        [
            data[:, i * current_stride : i * current_stride + window]
            for i in range(num_windows)
        ],
        dtype=np.float32,
    )

    
    


def plot_reconstruction(
    original,
    reconstructed,
    labels=None,
    channel=0,
    title=None,
    figsize=(14, 4)
):
    """
    Plot original vs reconstructed time series for visual inspection.

    Args:
        original (array): shape (T,) or (C, T)
        reconstructed (array): same shape as original
        labels (array, optional): anomaly labels (T,)
        channel (int): channel index if multivariate
        title (str, optional): plot title
    """
    orig = np.asarray(original)
    recon = np.asarray(reconstructed)

    # Handle multivariate
    if orig.ndim == 2:
        orig = orig[channel]
        recon = recon[channel]

    T = len(orig)
    t = np.arange(T)

    plt.figure(figsize=figsize)
    plt.plot(t, orig, label="Original", linewidth=2)
    plt.plot(t, recon, label="Reconstruction", linestyle="--")

    # Overlay anomaly regions if provided
    if labels is not None:
        labels = np.asarray(labels)
        for i in range(T):
            if labels[i] == 1:
                plt.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.05)

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
