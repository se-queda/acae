
global_config = {
    # --- Data ---
    "window_size": 64,
    "stride": 2,
    "batch_size": 64,
    "val_split": 0.2,

    # --- Optimizer ---
    "lr": 1e-4,  # learning rate

    # --- Preprocessing ---
    "savgol_len": 11,
    "savgol_poly": 3,
    "sparsity_factor": 10,
    "p_tile": 90,
    "mask_tiers": [95, 90, 85, 80],

    # --- Loss Weights ---
    "lambda_d": 1.0,        # discriminator
    "lambda_e": 1.0,        # encoder
    "recon_weight": 1.0,    # reconstruction
    "lambda_mom": 0.5,      # momentum
    "alpha_vpo": 150.0,     # Hamiltonian penalty
    "sentinel_weight": 10.0, #dead sensor penalty

    # --- Training ---
    "epochs": 20,
    "patience": 5,

    # --- Dataset ---
    "dataset": "SMD",
    "data_root": "/home/utsab/Downloads/smd/ServerMachineDataset/",
}

