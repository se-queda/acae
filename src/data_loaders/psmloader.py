import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Assuming your internal imports are still available
from src.router import route_features
from src.masking import get_masked_views, resi_masker
from src.utils import calculate_physics_jerk, create_spline_envelopes

def load_psm_windows(data_root, config):
    window = config['window_size']
    stride = config['stride']
    savgol_len = config['savgol_len']
    savgol_poly = config['savgol_poly']
    sparsity = config['sparsity_factor']

    print(f"🚀 Dual-Anchor Pipeline Initiated: PSM Dataset")
    
    # 1. Loading & Scaling (PSM CSVs)
    # PSM typically has a timestamp in the first column; we slice it out [:, 1:]
    train_df = pd.read_csv(os.path.join(data_root, "train.csv")).ffill().bfill()
    test_df = pd.read_csv(os.path.join(data_root, "test.csv")).ffill().bfill()
    label_df = pd.read_csv(os.path.join(data_root, "test_label.csv"))

    # Convert to values and drop the first column (timestamp)
    train_raw = train_df.values[:, 1:].astype(np.float32)
    test_raw = test_df.values[:, 1:].astype(np.float32)
    test_labels = label_df.values[:, 1:].flatten().astype(np.int32)

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
    
    # 5. Windowing
    def create_windows(data, current_stride):
        num_windows = (data.shape[0] - window) // current_stride + 1 
        return np.array([data[i*current_stride : i*current_stride + window] for i in range(num_windows)], dtype=np.float32)

    train_w_phy = create_windows(train_phy, stride)
    train_res_w = create_windows(train_res, stride)
    train_res_jerk_w = create_windows(total_res_jerk, stride)
    train_jerk_phy_w = create_windows(calculate_physics_jerk(train_phy, savgol_len, savgol_poly), stride)

    # 6. Masking Logic
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
        "phy": create_windows(test_phy, stride), 
        "res": create_windows(test_res, stride),
        "topology": topo
    }

    # 7. ACAE Label Slicing
    actual_test_len = (test_final["phy"].shape[0] - 1) * stride + window
    test_labels = test_labels[:actual_test_len]

    return train_final, test_final, test_labels, scaler.mean_, scaler.scale_