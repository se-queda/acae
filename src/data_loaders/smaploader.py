import os
import ast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import psutil

from src.configs.global_config import global_config
from src.router import route_features
from src.masking import multivariate_masker
from src.utils import calculate_physics_jerk, create_spline_envelopes, create_windows


def load_smap_windows(data_root, machine_id, config):
    """
    Specialized loader for SMAP (Soil Moisture Active Passive) spacecraft telemetry.
    output convention of each data view = (W, C, T)
    W = number of windows,
    C = number of channels / features / views,
    T = time steps per window
    """

    window = global_config["window_size"]
    stride = global_config["stride"]
    savgol_len = global_config["savgol_len"]
    savgol_poly = global_config["savgol_poly"]
    sparsity = global_config["sparsity_factor"]

    print(f"Dual-Anchor Pipeline Initiated: SMAP_{machine_id}")

    # 1.Input : raw npy (T, C)
    chan_id = machine_id.replace(".npy", "")
    train_path = os.path.join(data_root, "train", f"{chan_id}.npy")
    test_path = os.path.join(data_root, "test", f"{chan_id}.npy")

    train_raw = np.load(train_path).astype(np.float32)
    test_raw = np.load(test_path).astype(np.float32)

    print("After raw load")
    print("RAM:", psutil.Process(os.getpid()).memory_info().rss / 1e9, "GB")

    scaler = StandardScaler()
    train_total_norm = scaler.fit_transform(train_raw)  # (T, C_train)
    test_total_norm = scaler.transform(test_raw)  # (T, C_train)
    train_total_norm = train_total_norm.T  # (C_train, T)
    test_total_norm = test_total_norm.T  # (C_train, T)
    train_total_norm = train_total_norm.astype(np.float64, copy=False)
    test_total_norm = test_total_norm.astype(np.float64, copy=False)

    # Output: normalized arrays (C_train, T), (C_test, T)

    print("After normalising")
    print("RAM:", psutil.Process(os.getpid()).memory_info().rss / 1e9, "GB")

    # 2. Physics routing
    # Input : normalized signals (C_train, T), (C_test, T)
    (train_phy, train_res, test_phy, test_res), topo, phy_labels, res_labels = route_features(
        train_total_norm, test_total_norm
    )
    # Output: phy/res splits + cluster labels
    # train_phy : (C_phy, T)
    # train_res : (C_res, T)
    # test_phy : (C_phy, T)
    # test_res : (C_res, T)

    print("After routing")
    print("RAM:", psutil.Process(os.getpid()).memory_info().rss / 1e9, "GB")

    # 3. Residual envelopes
    # Input : residual signals (C_train, T_res)
    res_envelopes_upper = np.zeros_like(train_res)
    res_envelopes_lower = np.zeros_like(train_res)

    for local_idx in topo.res_to_lone_local:
        up, lo = create_spline_envelopes(
            train_res[local_idx, :],
            window,
            sparsity
        )
        res_envelopes_upper[local_idx, :] = up
        res_envelopes_lower[local_idx, :] = lo

    # Output: upper and lower envelopes of shape(C_train, T_res) each

    # 4. Isolated sensor jerk
    # Input : envelopes (C_envelop_upper, T_res), (C_envelop_lower, T_res)
    jerk_upper = calculate_physics_jerk(
        res_envelopes_upper, savgol_len, savgol_poly
    )
    jerk_lower = calculate_physics_jerk(
        res_envelopes_lower, savgol_len, savgol_poly
    )
    res_jerk = (jerk_upper + jerk_lower) / 2.0
    # Output: jerk signal (C_jerk, T_res)

    #5. Consensus Sensor jerk
    #input : (C_train, T_phy)
    train_phy_jerk = calculate_physics_jerk(
        train_phy, savgol_len, savgol_poly
    )
    #output: (C_jerk, T_phy)

    # 6. Jerk masking of conesnsus Features
    # Input : (C_train, T_phy)
    v1, v2, v3, v4 = multivariate_masker(
        train_phy, train_phy_jerk, phy_labels, use_consensus_masker=True
    )
    # Output: 4 masked views of shape (C_train, T_phy)

    #7. Jerk masking of isolated features
    r1 = multivariate_masker(
        train_res, res_jerk, res_labels, use_consensus_masker=False
    )  # (C_train, T_res)

    print("After masking")
    print("RAM:", psutil.Process(os.getpid()).memory_info().rss / 1e9, "GB")

    # 8. Sliding windows
    # Input : continuous signals (N_, F)
    train_w_phy = create_windows(train_phy, stride)  # (W, C_phy, T)
    train_phy_jerk = create_windows(train_phy_jerk, stride)  # (W, C_phy, T)
    v1 = create_windows(v1, stride)  # (W, C_phy,T)
    v2 = create_windows(v2, stride)  # (W, C_phy, T)
    v3 = create_windows(v3, stride)  # (W, C_phy, T)
    v4 = create_windows(v4, stride)  # (W, C_phy, T)
    r1 = create_windows(r1, stride)  # (W, C_res, T)
    # Output: windowed tensors of shape (W, C_phy, T) except for r1 its (W, C_res, T)

    # 9. Stacking all the data for consensus branch
    # Input : main view , 4 jerk views and jerk of (W, C_phy, T),
    # --------------------------------------------------
    phy_views = np.stack([train_w_phy, v1, v2, v3, v4, train_phy_jerk], axis=1)
    #Stacked, (W, 6, C_phy, T)

    print("After windowing")
    print("RAM:", psutil.Process(os.getpid()).memory_info().rss / 1e9, "GB")

    # --------------------------------------------------
    # 10. Final dictionaries
    # --------------------------------------------------
    train_final = {
        "phy_views": phy_views,
        "res_views": r1,
        "topology": topo
    }

    test_final = {
        "phy": create_windows(test_phy, stride),
        "res": create_windows(test_res, stride),
        "topology": topo
    }

    # 11. Dynamic Ground Truth Label Parsing (SMAP CSV)
    csv_path = os.path.join(data_root, "labeled_anomalies.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(data_root, "labelled_anomalies.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"SMAP Label CSV not found in {data_root}")

    df = pd.read_csv(csv_path)
    machine_info = df[df["chan_id"] == chan_id].iloc[0]

    num_test_steps = test_raw.shape[0]
    test_labels = np.zeros(num_test_steps, dtype=np.int32)
    anomaly_indices = ast.literal_eval(machine_info["anomaly_sequences"])
    for start, end in anomaly_indices:
        test_labels[start : end + 1] = 1

    # 12. Label Slicing : trunacted the last few points and its labels  that cant be in the window
    actual_test_len = (test_final["phy"].shape[0] - 1) * stride + window
    test_labels = test_labels[:actual_test_len]

    print("Final")
    print("RAM:", psutil.Process(os.getpid()).memory_info().rss / 1e9, "GB")

    return train_final, test_final, test_labels, scaler.mean_, scaler.scale_


if __name__ == "__main__":
    data_root = "/home/utsab/Downloads/SMAP"

    train_final, test_final, test_labels, _, _ = load_smap_windows(
        data_root=data_root,
        machine_id="A-1.npy",
        config=global_config,
    )
    train_final = train_final
    print("=== TRAIN ===")
    print("phy_views:", train_final["phy_views"].shape)   # (W, 6, C_phy, T)
    print("res_views:", train_final["res_views"].shape)   # (W, C_res, T)

    print("\n=== TEST ===")
    print("test_phy:", test_final["phy"].shape)           # (W, C_phy, T)
    print("test_res:", test_final["res"].shape)           # (W, C_res, T)
    print("labels:", test_labels.shape)

    # Verifier: recompute labels from CSV ranges and compare
    csv_path = os.path.join(data_root, "labeled_anomalies.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(data_root, "labelled_anomalies.csv")
    df = pd.read_csv(csv_path)
    chan_id = "A-1"
    machine_info = df[df["chan_id"] == chan_id].iloc[0]
    anomaly_indices = ast.literal_eval(machine_info["anomaly_sequences"])

    test_len = (test_final["phy"].shape[0] - 1) * global_config["stride"] + global_config["window_size"]
    expected_labels = np.zeros(test_len, dtype=np.int32)
    for start, end in anomaly_indices:
        expected_labels[start : end + 1] = 1
    expected_labels = expected_labels[: test_labels.shape[0]]

    match = np.array_equal(test_labels, expected_labels)
    diff = int(np.sum(test_labels != expected_labels))
    print("Label match:", match)
    print("Label diffs:", diff)
