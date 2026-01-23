import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import math
import pandas as pd

window_size= 100
def mix_features(z_orig, z_other):
    """
    Interpolates between anchor and augmented latents for the discriminator task.
    """
    batch_size = tf.shape(z_orig)[0]
    alpha = tf.random.uniform((batch_size, 1), minval=0.0, maxval=1.0)
    z_mixed = alpha * z_orig + (1.0 - alpha) * z_other
    return z_mixed, alpha


def create_windows(jerk):
    T = len(jerk)
    windows = []
    
    for start in range(0, T, window_size):
        end = min(start+window_size, T)
        windows.append(jerk[start:end])
    return windows

def calc_budget(jerk, tile):
    threshold = np.percentile(jerk, tile)
    count = np.sum(jerk > threshold)
    budget = math.ceil(count/window_size)
    if count> 0 and budget == 0:
        budget = 1
    return budget

def project_to_time(mask_windows, Time):
    time_mask = np.zeros(Time, dtype = bool)
    
    for w in mask_windows:
        start = w * window_size
        end = min(start + window_size, Time)
        time_mask[start:end] = True
    return time_mask

def apply_mask(feature, time_mask):
    masked_feature = feature.copy()
    masked_feature[time_mask] = 0.0
    return masked_feature
        
    
def univariate_masker(feature, jerk, tile):
    T = len(feature)
    jerk_windows = create_windows(jerk)
    jerk_window_mean =[w.mean() for w in jerk_windows]
    rankedwindow = np.argsort(jerk_window_mean)[::-1]
    budget = calc_budget(jerk, tile)
    windows_to_mask = rankedwindow[:budget]
    time_mask = project_to_time(
        windows_to_mask,T
    )
    masked_feature = apply_mask(feature, time_mask)
    
    return masked_feature
    
def consensus_masker(data, masked_data, cluster_labels):
    T, F = data.shape

    data_np = data.values
    masked_np = masked_data.values


    data_windows = create_windows(data_np)       
    masked_windows = create_windows(masked_np)   

    final_windows = []
    unique_clusters = np.unique(cluster_labels)

    for w_masked, w_orig in zip(masked_windows, data_windows):
        
        w_consensus = w_masked.copy()
        for c_id in unique_clusters:
            c_indices = np.where(cluster_labels == c_id)[0]
            feature_masked = np.all(
                w_masked[:, c_indices] == 0.0,
                axis=0
            )  # 

            if not np.all(feature_masked):
                w_consensus[:, c_indices] = w_orig[:, c_indices]

        final_windows.append(w_consensus)

    consensus_np = np.vstack(final_windows)[:T]

    return pd.DataFrame(consensus_np, columns=data.columns)

def multivariate_masker(data, jerk, cluster_labels, use_consensus_masker = True):
    if use_consensus_masker:
        tiers = [95, 90, 85, 80]
    else: 
        tiers = [90]
    
    views = []
    for tier in tiers:
        masked_df = data.copy()
        for col in data.columns:
            signal = data[col].values
            jerk_data = jerk[col].values
            masked_df[col] = univariate_masker(signal,jerk_data, tier)
        if use_consensus_masker:
            consensus_df = consensus_masker(data , masked_df, cluster_labels)
        else :
            consensus_df = masked_df
        
        views.append(consensus_df)
    return tuple(views)

