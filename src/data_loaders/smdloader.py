import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from global_config import global_config

# Assuming your internal imports are still available
from src.router import route_features
from src.masking import multivariate_masker
from src.utils import calculate_physics_jerk, create_spline_envelopes, load_file, create_windows



# input = (raw text/csv/xlsx/npy file of size T, C), loads raw data from local machine.
def load_smd_windows(data_root, machine_id, config):
    """
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

    print(f"Dual-Anchor Pipeline Initiated: {machine_id}")


    # 1.Input : raw txt (T, C)
    train_raw = load_file(
        os.path.join(data_root, "train", f"{machine_id}.txt")
    )
    test_raw = load_file(
        os.path.join(data_root, "test", f"{machine_id}.txt")
    )
    test_labels = load_file(
        os.path.join(data_root, "test_label", f"{machine_id}.txt")
    ).flatten().astype(np.int32)

    scaler = StandardScaler()
    train_total_norm = scaler.fit_transform(train_raw) #(T, C_train)
    test_total_norm = scaler.transform(test_raw)#(T, C_train)
    train_total_norm = train_total_norm.T   # (C_train, T)
    test_total_norm  = test_total_norm.T    # (C_train, T)       
    # Output: normalized arrays (C_train, T), (C_test, T)


    # Physics routing
    # Input : normalized signals (C_train, T), (C_test, T)
    (train_phy, train_res,test_phy, test_res), topo, phy_labels, res_labels= route_features(
        train_total_norm,
        test_total_norm
    )
    # Output: phy/res splits + cluster labels
    # train_phy : (C_phy, T)
    # train_res : (C_res, T)
    # test_phy : (C_phy, T)
    # test_res : (C_res, T)



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

    v1, v2, v3, v4 = multivariate_masker(train_phy,train_phy_jerk,phy_labels,use_consensus_masker=True)
    # Output: 4 masked views of shape (C_train, T_phy)
    
    #7. Jerk masking of isolated features
    r1 = multivariate_masker(train_res,res_jerk,res_labels,use_consensus_masker=False) #(C_train, T_res)

    
    # 8. Sliding windows
    # Input : continuous signals (N_, F)
    train_w_phy = create_windows(train_phy, stride)  # (W, C_phy, T)
    train_phy_jerk = create_windows(train_phy_jerk, stride) # (W, C_phy, T)
    v1 = create_windows(v1,stride) # (W, C_phy,T)                               
    v2 = create_windows(v2,stride) # (W, C_phy, T)
    v3 = create_windows(v3,stride) # (W, C_phy, T)
    v4 = create_windows(v4,stride) # (W, C_phy, T)
    r1 = create_windows(r1,stride) # (W, C_res, T)
    # Output: windowed tensors of shape (W, C_phy, T) except for r1 its (W, C_res, T)

    # 9. Stacking all the data for consensus branch
    # Input : main view , 4 jerk views and jerk of (W, C_phy, T), 
    # --------------------------------------------------
    phy_views = np.stack([train_w_phy,v1, v2, v3, v4,train_phy_jerk],axis=1)
    #Stacked, (W, 6, C_phy, T)  


    # --------------------------------------------------
    # 10. Final dictionaries
    # --------------------------------------------------
    train_final = {
        "phy_views": phy_views,     # Final consensus branch training data going inside dataset_builder.py 
        "res_views": r1,            # Final isolated training data going inside dataset_builder.py
        "topology": topo
    }

    test_final = {
        "phy": create_windows(test_phy, stride), #Final consensus branch test data going inside dataset_builder.py 
        "res": create_windows(test_res, stride),  #Final isolated branch test data going inside dataset_builder.py 
        "topology": topo
    }



    # 11. Label Slicing : trunacted the last few points and its labels  that cant be in the window
    actual_test_len = (test_final["phy"].shape[0] - 1) * stride + window
    test_labels = test_labels[:actual_test_len]

    return train_final, test_final, test_labels



