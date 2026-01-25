import numpy as np
from sklearn.cluster import AgglomerativeClustering

class MachineTopology:
    def __init__(self, idx_phy, idx_res, idx_lone, idx_dead):
        # 1. Global Indices (Original Sensor IDs 0-37)
        self.idx_phy = np.array(idx_phy)
        self.idx_res = np.array(idx_res)
        self.idx_lone = np.array(idx_lone)
        self.idx_dead = np.array(idx_dead)
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
        print(f"HNN Branch : {len(self.idx_phy)} features")
        print(f"FNO branch  : {len(self.idx_res)} total")
        print(f"  └─ Isolated sensors      : {len(self.idx_lone)}")
        print(f"  └─ Dead Sensors   : {len(self.idx_dead)}")
        print("-" * 35)



def dead_sensor_finder(train_data):
    
    # train_data: (C, T) -> variance per sensor
    variances = np.var(train_data, axis = 1)
    idx_dead = np.where(variances < 1e-9)[0]
    return idx_dead


def cluster_finder(data, dist_threshold):
    # data: (C, T) -> correlate sensors (rows)
    corr = np.nan_to_num(np.corrcoef(data, rowvar = True))
    
    dist = 1 - np.abs(corr)
    if dist.ndim == 0:
        dist = dist.reshape(1,1)
        
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=dist_threshold, 
        linkage='complete'
    )
    labels = clustering.fit_predict(dist)
    
    return labels
    
    
def route_features(train_data, test_data, dist_threshold=0.5):
    # Expect (C, T) always
    num_total_features = train_data.shape[0]
    all_indices = np.arange(num_total_features)
    
    idx_dead = dead_sensor_finder(train_data)
    idx_active = np.setdiff1d(all_indices, idx_dead)

    if len(idx_active) < 2:
        print(f" Features with no correlated sesnsors detected, only FNO path will be used")
        idx_phy = np.array([], dtype=int)
        idx_lone = idx_active
        idx_res = all_indices
        phy_cluster_labels = np.array([], dtype=int)
        res_labels = np.arange(len(idx_res), dtype=int)
        
        topo = MachineTopology(idx_phy, idx_res, idx_lone, idx_dead)
        topo.summary()
        
        return (train_data[:, :0], train_data, test_data[:, :0], test_data), topo, phy_cluster_labels, res_labels


    train_active = train_data[idx_active, :]
    labels= cluster_finder(train_active, dist_threshold)
    
    

    unique_ids, counts = np.unique(labels, return_counts = True)
    consensus_ids = unique_ids[counts>1]
    is_consensus = np.isin(labels, consensus_ids)

    phy_cluster_labels = labels[is_consensus]
    idx_phy = idx_active[is_consensus]
    idx_lone = idx_active[~is_consensus]

    idx_res = np.sort(np.concatenate([idx_lone, idx_dead]))
    res_labels = np.arange(len(idx_res), dtype=int)

    topo = MachineTopology(idx_phy, idx_res, idx_lone, idx_dead)
    topo.summary()


    train_phy, train_res = train_data[idx_phy, :], train_data[idx_res, :]
    test_phy, test_res = test_data[idx_phy, :], test_data[idx_res, :]

    return (train_phy, train_res, test_phy, test_res), topo, phy_cluster_labels, res_labels
