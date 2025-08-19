import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


def load_txt_file(path):
    return np.loadtxt(path, delimiter=',', dtype=np.float32)


def load_smd_windows(data_root, machine_id, window=64, stride=2):
    """
    Load and window the SMD dataset for a specific machine.
    """
    # Load raw data
    train_data = load_txt_file(f"{data_root}/train/{machine_id}.txt")
    test_data = load_txt_file(f"{data_root}/test/{machine_id}.txt")
    test_labels = load_txt_file(f"{data_root}/test_label/{machine_id}.txt").flatten()

    # Normalize using train stats
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    def create_windows(data):
        return np.array([
            data[i:i + window]
            for i in range(0, data.shape[0] - window + 1, stride)
        ], dtype=np.float32)

    # Create sliding windows
    train_w = create_windows(train_data)
    test_w = create_windows(test_data)
    y_test_win = np.array([
        int(np.any(test_labels[i:i + window]))
        for i in range(0, len(test_labels) - window + 1, stride)
    ])

    return train_w, test_w, y_test_win, scaler.mean_, scaler.scale_


def build_tf_datasets(train_w, test_w, val_split=0.2, batch_size=128):
    """
    Convert numpy windows into tf.data.Datasets.
    """
    num_train = int(len(train_w) * (1 - val_split))
    train_data, val_data = train_w[:num_train], train_w[num_train:]

    AUTOTUNE = tf.data.AUTOTUNE

    def make_dataset(data, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices(data)
        if shuffle:
            ds = ds.cache().shuffle(2048)
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    train_ds = make_dataset(train_data, shuffle=True)
    val_ds = make_dataset(val_data)
    test_ds = make_dataset(test_w)

    return train_ds, val_ds, test_ds
