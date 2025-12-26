import numpy as np
import os
from src.utils import load_smd_windows

def scan_dead_features(machine_id, data_root, threshold=1e-6):
    """
    Scans for sensors with near-zero variance (Dead Features).
    """
    print(f"🔍 Scanning {machine_id} for Dead Features...")
    
    # 1. Load Data (Standard SMD Loading)
    _, test_final, _, _, _ = load_smd_windows(
        data_root=data_root,
        machine_id=machine_id,
        window=64
    )
    
    # 2. Extract Raw Feature Matrix
    # We look at the 'res' windows because they contain the full sensor set
    res_data = test_final.get('res', test_final.get('res_windows'))
    
    # Reshape from (N, Window, Features) to (N*Window, Features) to get global stats
    flat_data = res_data.reshape(-1, res_data.shape[-1])
    
    # 3. Calculate Variance per Sensor
    variances = np.var(flat_data, axis=0)
    means = np.mean(flat_data, axis=0)
    
    # 4. Identify Dead Sensors
    dead_indices = np.where(variances < threshold)[0]
    
    print("-" * 30)
    print(f"📊 Scan Results for {machine_id}:")
    print(f"Total Sensors Checked: {flat_data.shape[-1]}")
    print(f"Dead Sensors Detected: {len(dead_indices)}")
    
    if len(dead_indices) > 0:
        print("\nIndex | Mean Value | Variance")
        for idx in dead_indices:
            print(f"{idx:5} | {means[idx]:10.4f} | {variances[idx]:10.8f}")
    
    # Cross-reference with your Topology if available
    topo = test_final.get('topology')
    if topo:
        expected_dead = set(topo.res_to_dead_local)
        detected_dead = set(dead_indices)
        
        matches = expected_dead.intersection(detected_dead)
        missed = expected_dead - detected_dead
        ghosts = detected_dead - expected_dead
        
        print("\n--- Topology Cross-Check ---")
        print(f"✅ Correctly Masked: {len(matches)}")
        if missed: print(f"⚠️ Topo says Dead, but data is Active: {list(missed)}")
        if ghosts: print(f"🚫 Data is Dead, but Topo says Active: {list(ghosts)}")
    
    return dead_indices

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    SMD_PATH = "/home/utsab/Downloads/smd/ServerMachineDataset"
    MID = "machine-1-1"
    
    dead_feats = scan_dead_features(MID, SMD_PATH)