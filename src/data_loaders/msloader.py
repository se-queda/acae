import os
import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import StandardScaler

# Internal imports
from src.router import route_features, MachineTopology
from src.masking import get_masked_views, resi_masker
from src.utils import calculate_physics_jerk, create_spline_envelopes

def load_msl_windows(data_root, machine_id, config):
    """
    Robust loader for MSL telemetry. 
    Handles missing CSV entries and 0-feature HNN routing.
    """
    window = config['window_size'] 
    stride = config['stride'] 
    savgol_len = config['savgol_len']
    savgol_poly = config['savgol_poly']
    sparsity = config['sparsity_factor']

    print(f"🚀 Dual-Anchor Pipeline Initiated: MSL_{machine_id}")
    
    # 1. Loading & Standardization
    train_path = os.path.join(data_root, "train", f"{machine_id}.npy")
    test_path = os.path.join(data_root, "test", f"{machine_id}.npy")
    
    train_raw = np.load(train_path).astype(np.float32)
    test_raw = np.load(test_path).astype(np.float32)
    
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_raw)
    test_norm = scaler.transform(test_raw)

    # 2. Routing with Minimum Feature Guard
    (train_phy, train_res, test_phy, test_res), topo, phy_labels = route_features(train_norm, test_norm)


    # cluster labels required by consensus_masker — single cluster id is enough
    phy_labels = np.zeros(train_phy.shape[1], dtype=int)

    # 3. Envelopes & Jerk
    res_envelopes_upper = np.zeros_like(train_res)
    res_envelopes_lower = np.zeros_like(train_res)
    for local_idx in topo.res_to_lone_local:
        up, lo = create_spline_envelopes(train_res[:, local_idx], window, sparsity)
        res_envelopes_upper[:, local_idx] = up
        res_envelopes_lower[:, local_idx] = lo

    j_up = calculate_physics_jerk(res_envelopes_upper, savgol_len, savgol_poly)
    j_lo = calculate_physics_jerk(res_envelopes_lower, savgol_len, savgol_poly)
    total_res_jerk = (j_up + j_lo) / 2.0
    
    # 4. Windowing
    def create_windows(data, current_stride):
        num_windows = (data.shape[0] - window) // current_stride + 1 
        return np.array([data[i*current_stride : i*current_stride + window] for i in range(num_windows)], dtype=np.float32)

    train_w_phy = create_windows(train_phy, stride)
    train_res_w = create_windows(train_res, stride)
    train_res_jerk_w = create_windows(total_res_jerk, stride)
    train_jerk_phy_w = create_windows(calculate_physics_jerk(train_phy, savgol_len, savgol_poly), stride)
    
    # 5. Masking Views
    v1, v2, v3, v4 = get_masked_views(train_w_phy, train_jerk_phy_w, phy_labels)
    phy_views = np.stack([train_w_phy, v1, v2, v3, v4, train_jerk_phy_w], axis=1)
    rv1 = resi_masker(train_res_w, train_res_jerk_w, p_tile=config['p_tile'])

    train_final = {"phy_views": phy_views, "res_views": rv1, "topology": topo}
    test_final = {
        "phy": create_windows(test_phy, stride),
        "res": create_windows(test_res, stride), 
        "topology": topo
    }

    # 6. Robust CSV Label Parsing (The INDEX FIX)
    csv_path = os.path.join(data_root, "labeled_anomalies.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(data_root, "labelled_anomalies.csv")
        
    df = pd.read_csv(csv_path)
    
    # Search with explicit query to avoid .iloc[0] on empty results
    query = df[df['chan_id'] == machine_id]
    test_labels = np.zeros(test_raw.shape[0], dtype=np.int32)

    if query.empty:
        print(f"❌ Warning: {machine_id} not found in {csv_path}. Proceeding with 0-labels.")
    else:
        machine_info = query.iloc[0]
        # NASA column name is 'anomaly_sequences'
        try:
            anomaly_indices = ast.literal_eval(machine_info['anomaly_sequences'])
            for start, end in anomaly_indices:
                # NASA indices are inclusive [start, end]
                test_labels[start : end + 1] = 1
        except:
            print(f"⚠️ Failed to parse anomaly sequences for {machine_id}")
    
    # 7. Paper-Aligned Slicing
    actual_test_len = (test_final["phy"].shape[0] - 1) * stride + window
    test_labels = test_labels[:actual_test_len]

    return train_final, test_final, test_labels, scaler.mean_, scaler.scale_