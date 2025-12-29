import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ast
from src.router import route_features
from src.masking import get_masked_views, resi_masker
from src.utils import calculate_physics_jerk, create_spline_envelopes

def load_smap_windows(data_root, machine_id, config):
    """
    Specialized loader for SMAP (Soil Moisture Active Passive) spacecraft telemetry.
    Processes 25 dimensions and parses labels from a central CSV file.
    """
    window = config['window_size'] 
    stride = config['stride'] 
    savgol_len = config['savgol_len']
    savgol_poly = config['savgol_poly']
    sparsity = config['sparsity_factor']

    print(f"🚀 Dual-Anchor Pipeline Initiated: SMAP_{machine_id}")
    
    # [cite_start]1. Loading & Standardization [cite: 234, 273]
    train_path = os.path.join(data_root, "train", f"{machine_id}.npy")
    test_path = os.path.join(data_root, "test", f"{machine_id}.npy")
    
    train_raw = np.load(train_path).astype(np.float32)
    test_raw = np.load(test_path).astype(np.float32)
    
    scaler = StandardScaler()
    train_total_norm = scaler.fit_transform(train_raw)
    test_total_norm = scaler.transform(test_raw)

    # 2. Physical Routing (Dual-Anchor Topology)
    (train_phy, train_res, test_phy, test_res), topo, phy_labels = route_features(
        train_total_norm, test_total_norm
    )
    
    # 3. Lone Wolf Envelope Generation (Residual Branch)
    res_envelopes_upper = np.zeros_like(train_res)
    res_envelopes_lower = np.zeros_like(train_res)

    for local_idx in topo.res_to_lone_local:
        up, lo = create_spline_envelopes(train_res[:, local_idx], window, sparsity)
        res_envelopes_upper[:, local_idx] = up
        res_envelopes_lower[:, local_idx] = lo

    # 4. Physics Jerk Calculation (Noise Floor)
    j_up = calculate_physics_jerk(res_envelopes_upper, savgol_len, savgol_poly)
    j_lo = calculate_physics_jerk(res_envelopes_lower, savgol_len, savgol_poly)
    total_res_jerk = (j_up + j_lo) / 2.0
    
    
    def create_windows(data, current_stride):
        num_windows = (data.shape[0] - window) // current_stride + 1 
        return np.array([data[i*current_stride : i*current_stride + window] for i in range(num_windows)], dtype=np.float32)

    train_w_phy = create_windows(train_phy, stride)
    train_res_w = create_windows(train_res, stride)
    train_res_jerk_w = create_windows(total_res_jerk, stride)
    train_jerk_phy_w = create_windows(calculate_physics_jerk(train_phy, savgol_len, savgol_poly), stride)

    # [cite_start]6. Multi-Scale Masking (ACAE Proxy Task) [cite: 26, 300, 327]
    from src.masking import get_masked_views, resi_masker
    v1, v2, v3, v4 = get_masked_views(train_w_phy, train_jerk_phy_w, phy_labels)
    phy_views = np.stack([train_w_phy, v1, v2, v3, v4, train_jerk_phy_w], axis=1) 
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

# 7. Dynamic Ground Truth Label Parsing
    # The summary confirms the file is likely 'labeled_anomalies.csv'
    csv_path = os.path.join(data_root, "labeled_anomalies.csv")
    if not os.path.exists(csv_path):
        # Fallback for double 'l' spelling just in case
        csv_path = os.path.join(data_root, "labelled_anomalies.csv")
        
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"SMAP Label CSV not found in {data_root}")
        
    df = pd.read_csv(csv_path)
    
    # 1. Identify row using 'chan_id' column
    machine_info = df[df['chan_id'] == machine_id].iloc[0]
    
    # 2. Initialize mask based on the actual test length
    num_test_steps = test_raw.shape[0]
    test_labels = np.zeros(num_test_steps, dtype=np.int32)
    
    # 3. Use 'anomaly_sequences' column as shown in your summary
    # This column contains the list of start & end timestamps
    anomaly_indices = ast.literal_eval(machine_info['anomaly_sequences'])
    
    for start, end in anomaly_indices:
        # Paint the anomaly mask (inclusive of end index)
        test_labels[start : end + 1] = 1
    
    # 4. Slicing labels to match windowed sequence length
    actual_test_len = (test_final["phy"].shape[0] - 1) * stride + window
    test_labels = test_labels[:actual_test_len]

    return train_final, test_final, test_labels, scaler.mean_, scaler.scale_