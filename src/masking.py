import torch
import numpy as np
import math
import pandas as pd

window_size= 100
def mix_features(z_orig, z_other):
    """
    Interpolates between anchor and augmented latents for the discriminator task.
    """
    batch_size = z_orig.shape[0]
    alpha = torch.rand(batch_size, 1).to(z_orig.device)
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
    C, T = data.shape

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
                axis = 1
            )  

            if not np.all(feature_masked):
                w_consensus[:, c_indices] = w_orig[:, c_indices]

        final_windows.append(w_consensus)

    consensus = np.concatenate(final_windows, axis=1)[:, :T]

    return consensus

def multivariate_masker(data, jerk, cluster_labels, use_consensus_masker=True):
    if use_consensus_masker:
        tiers = [95, 90, 85, 80]
    else:
        tiers = [90]

    C, T = data.shape
    views = []

    for tier in tiers:

        masked = data.copy()
        for c in range(C):
            masked[c] = univariate_masker(
                feature=data[c],
                jerk=jerk[c],
                tile=tier
            )
        if use_consensus_masker:
            consensus = consensus_masker(
                data, masked, cluster_labels
            )
        else:
            consensus = masked

        views.append(consensus)

    return tuple(views)


