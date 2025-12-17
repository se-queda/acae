import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os

def load_txt_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return np.loadtxt(path, delimiter=',', dtype=np.float32)
    except:
        return np.loadtxt(path, dtype=np.float32)

def load_smd_windows(data_root, machine_id, window=64, train_stride=1):
    """
    Loads SMD data and prepares it for Point-wise Anomaly Detection (Paper Sec 3.1).
    
    Args:
        train_stride (int): Overlap for training data (default 1 for max augmentation).
        
    Returns:
        train_w: Windowed training data (for model input).
        test_w:  Windowed testing data (Non-overlapping for easy reconstruction).
        test_labels: RAW point-wise labels (for paper-style evaluation).
    """
    # 1. Load raw point-wise data
    print(f"Loading {machine_id} from {data_root}...")
    train_path = os.path.join(data_root, "train", f"{machine_id}.txt")
    test_path = os.path.join(data_root, "test", f"{machine_id}.txt")
    label_path = os.path.join(data_root, "test_label", f"{machine_id}.txt")

    train_data = load_txt_file(train_path)
    test_data = load_txt_file(test_path)
    test_labels = load_txt_file(label_path).flatten() # Shape: (T,)

    # 2. Normalize (Fit on Train, Apply to Test)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    def create_windows(data, stride):
        # Drop the 'tail' data that doesn't fit into a full window
        num_windows = (data.shape[0] - window) // stride + 1
        return np.array([
            data[i*stride : i*stride + window]
            for i in range(num_windows)
        ], dtype=np.float32)

    # 3. Create Windows
    # Train: Overlapping (stride=1 usually) to maximize learning samples
    train_w = create_windows(train_data, stride=train_stride)
    
    # Test: Non-overlapping (stride=window) so we can simply .flatten() the reconstruction
    # to compare against the original point-wise labels.
    test_w = create_windows(test_data, stride=window)
    
    # Trim labels to match the exact length covered by the test windows
    # Since we dropped the tail in 'create_windows', we must drop the tail in labels too.
    covered_length = test_w.shape[0] * window
    test_labels = test_labels[:covered_length]

    print(f"Data Loaded: Train Windows: {train_w.shape}, Test Windows: {test_w.shape}")
    print(f"Test Labels Point-wise Shape: {test_labels.shape}")

    return train_w, test_w, test_labels, scaler.mean_, scaler.scale_

def build_tf_datasets(train_w, test_w, val_split=0.2, batch_size=128):
    # Split Train into Train/Val
    num_train = int(len(train_w) * (1 - val_split))
    
    # Shuffle indices for random split (better than sequential split for generic training)
    indices = np.arange(len(train_w))
    np.random.shuffle(indices)
    
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]
    
    train_data = train_w[train_idx]
    val_data = train_w[val_idx]

    AUTOTUNE = tf.data.AUTOTUNE

    def make_dataset(data, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices(data)
        if shuffle:
            ds = ds.cache().shuffle(2048)
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    train_ds = make_dataset(train_data, shuffle=True)
    val_ds = make_dataset(val_data, shuffle=False)
    test_ds = make_dataset(test_w, shuffle=False) # Never shuffle test set!

    return train_ds, val_ds, test_ds

