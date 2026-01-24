import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DualAnchorDataset(Dataset):
    """
    Dataset yielding (phy_view, res_view) pairs.
    Shapes are assumed to be (C, T) per sample.
    """
    def __init__(self, phy_data, res_data):
        assert len(phy_data) == len(res_data)
        self.phy = torch.as_tensor(phy_data, dtype=torch.float32)
        self.res = torch.as_tensor(res_data, dtype=torch.float32)

    def __len__(self):
        return self.phy.shape[0]

    def __getitem__(self, idx):
        # Returns (C, T), (C, T)
        return self.phy[idx], self.res[idx]


#input = ()
    
def dataloader(train_final,test_final,val_split=0.2,batch_size=128,num_workers=4,):


    phy_data = train_final["phy_views"]   # (N, C_phy, T)
    res_data = train_final["res_views"]   # (N, C_res, T)
    num_samples = len(phy_data)

    # --- random window-level split ---
    indices = np.random.permutation(num_samples)
    num_train = int(num_samples * (1 - val_split))

    train_idx = indices[:num_train]
    val_idx = indices[num_train:]

    train_phy = phy_data[train_idx]
    train_res = res_data[train_idx]

    val_phy = phy_data[val_idx]
    val_res = res_data[val_idx]

    test_phy = test_final["phy"]   # (N_test, C_phy, T)
    test_res = test_final["res"]   # (N_test, C_res, T)

    # --- datasets ---
    train_ds = DualAnchorDataset(train_phy, train_res)
    val_ds   = DualAnchorDataset(val_phy, val_res)
    test_ds  = DualAnchorDataset(test_phy, test_res)

    # --- dataloaders ---
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,      # TF shuffle
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


"""""
# ---- dummy dimensions ----
N = 100        # number of windows
C_phy = 4      # physics channels
C_res = 6      # residual channels
T = 64         # timesteps

# ---- fake training data ----
train_final = {
    "phy_views": np.random.randn(N, C_phy, T).astype(np.float32),
    "res_views": np.random.randn(N, C_res, T).astype(np.float32),
}

# ---- fake test data ----
test_final = {
    "phy": np.random.randn(20, C_phy, T).astype(np.float32),
    "res": np.random.randn(20, C_res, T).astype(np.float32),
}

train_loader, val_loader, test_loader = build_torch_dataloaders(
    train_final,
    test_final,
    val_split=0.2,
    batch_size=16,
    num_workers=0,   # use 0 for quick local testing
)

phy_batch, res_batch = next(iter(train_loader))

print("Train batch:")
print("phy shape:", phy_batch.shape)
print("res shape:", res_batch.shape)

assert phy_batch.shape == (16, C_phy, T)
assert res_batch.shape == (16, C_res, T)
"""
