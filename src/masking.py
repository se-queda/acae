import tensorflow as tf
from tensorflow.keras import layers

# Projection layer used in ACAE (optional, used in contrastive evals if needed)
projection_layer = layers.Dense(256)


def mix_features(z_orig, z_other):
    """
    Mix latent vectors with random alpha ∈ [0,1] for contrastive learning.
    """
    batch_size = tf.shape(z_orig)[0]
    alpha = tf.random.uniform((batch_size, 1), minval=0.0, maxval=1.0)
    z_mixed = alpha * z_orig + (1.0 - alpha) * z_other
    return z_mixed, alpha


def generate_masked_views(X, mask_rates):
    """
    Generate masked versions of input X using multiple mask rates.
    """
    batch_size = tf.shape(X)[0]
    time_steps = X.shape[1]  # Static shape for XLA
    views = []

    for rate in mask_rates:
        n_mask = int(round(rate * time_steps))

        def mask_one(_):
            idx = tf.random.shuffle(tf.range(time_steps))[:n_mask]
            return tf.tensor_scatter_nd_update(
                tf.ones(time_steps), tf.expand_dims(idx, 1), tf.zeros(n_mask)
            )

        # ✅ Use fn_output_signature to avoid deprecation
        mask = tf.map_fn(
            mask_one,
            tf.range(batch_size),
            fn_output_signature=tf.TensorSpec([time_steps], dtype=tf.float32)
        )

        masked = X * tf.expand_dims(mask, -1)
        views.append(masked)

    return views
