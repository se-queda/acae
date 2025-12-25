import numpy as np
from sklearn.cluster import AgglomerativeClustering

class MachineTopology:
    """
    Physical Address Table for the Dual-Anchor Engine.
    Maps local branch columns back to global sensor indices.
    """
    def __init__(self, idx_phy, idx_res, idx_lone, idx_dead):
        # 1. Global Indices (Original Sensor IDs 0-37)
        self.idx_phy = np.array(idx_phy)
        self.idx_res = np.array(idx_res)
        self.idx_lone = np.array(idx_lone)
        self.idx_dead = np.array(idx_dead)
        
        # 2. Local Branch Mapping: 
        # Locates where specific global sensors are sitting inside the branch matrices
        self.res_to_dead_local = [
            np.where(self.idx_res == gid)[0][0] for gid in self.idx_dead
        ]
        self.res_to_lone_local = [
            np.where(self.idx_res == gid)[0][0] for gid in self.idx_lone
        ]

    def summary(self):
        print("-" * 35)
        print(f"🛠️  MACHINE TOPOLOGY BLUEPRINT")
        print("-" * 35)
        print(f"Consensus Branch (HNN) : {len(self.idx_phy)} features")
        print(f"Residual Branch (AE)  : {len(self.idx_res)} total")
        print(f"  └─ Lone Wolves      : {len(self.idx_lone)}")
        print(f"  └─ Dead Sentinels   : {len(self.idx_dead)}")
        print("-" * 35)

def route_features(train_data, test_data, dist_threshold=0.5):
    """
    The Brain of the Pipeline:
    Splits features into 'System Physics' and 'Residual Sentinel' groups.
    """
    num_total_features = train_data.shape[1]
    all_indices = np.arange(num_total_features)

    # --- 1. Dead Sentinel Extraction ---
    # Sensors that never move in training are perfect zero-anchors
    variances = np.var(train_data, axis=0)
    idx_dead = np.where(variances < 1e-9)[0]
    
    idx_active = np.setdiff1d(all_indices, idx_dead)
    train_active = train_data[:, idx_active]

    # --- 2. Consensus Discovery ---
    # Group active sensors that move in harmony (The System Orchestra)
    corr = np.nan_to_num(np.corrcoef(train_active, rowvar=False))
    dist = 1 - np.abs(corr)
    
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=dist_threshold, 
        linkage='complete'
    )
    labels = clustering.fit_predict(dist)

    # --- 3. Cluster Filtering ---
    # Only keep groups of size > 1 for the HNN branch
    u_ids, counts = np.unique(labels, return_counts=True)
    consensus_ids = u_ids[counts > 1]
    is_consensus = np.isin(labels, consensus_ids)
    
    # Extract specific labels for the Physics branch (Required for Masker Veto)
    phy_cluster_labels = labels[is_consensus]

    # --- 4. Final Routing & Topology ---
    idx_phy = idx_active[is_consensus]
    idx_lone = idx_active[~is_consensus]
    idx_res = np.sort(np.concatenate([idx_lone, idx_dead]))

    topo = MachineTopology(idx_phy, idx_res, idx_lone, idx_dead)
    topo.summary()

    # --- 5. Data Splitting ---
    train_phy, train_res = train_data[:, idx_phy], train_data[:, idx_res]
    test_phy, test_res = test_data[:, idx_phy], test_data[:, idx_res]

    # Return everything needed to feed the model and the masker
    return (train_phy, train_res, test_phy, test_res), topo, phy_cluster_labels