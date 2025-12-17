import tensorflow as tf
from tensorflow.keras import layers, models

def bottleneck_block(filters, stride=1, expansion=4, is_decoder=False):
    """
    Implements the Bottleneck Residual Block from Fig. 5.
    Structure: 1x1 Conv -> 3x3 Conv -> 1x1 Conv
    Activations: Tanh after BN (as per diagram)
    """
    expanded_filters = filters * expansion

    def block(x_input):
        # --- Shortcut Path ---
        shortcut = x_input
        if stride > 1 or x_input.shape[-1] != expanded_filters:
            if is_decoder:
                # Decoder upsamples in the shortcut
                shortcut = layers.UpSampling1D(size=stride)(shortcut)
                shortcut = layers.Conv1D(expanded_filters, 1, padding='same')(shortcut)
            else:
                # Encoder downsamples in the shortcut
                shortcut = layers.Conv1D(expanded_filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # --- Main Path ---
        # 1. 1x1 Conv (Reduce / Projection)
        x = layers.Conv1D(filters, 1, padding='same')(x_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('tanh')(x)

        # 2. 3x3 Conv (Spatial Processing)
        if is_decoder and stride > 1:
            x = layers.UpSampling1D(size=stride)(x)
            x = layers.Conv1D(filters, 3, padding='same')(x)
        else:
            x = layers.Conv1D(filters, 3, strides=stride, padding='same')(x)
        
        x = layers.BatchNormalization()(x)
        x = layers.Activation('tanh')(x)

        # 3. 1x1 Conv (Expand)
        x = layers.Conv1D(expanded_filters, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)

        # --- Add & Final Activation ---
        out = layers.Add()([x, shortcut])
        return layers.Activation('tanh')(out)

    return block

def build_projection_head(input_shape=(64, 38)):
    """
    Matches "Projection" block in Fig. 5:
    Conv 1x1, 128 -> Conv 1x3, 128 -> Conv 1x7, 128 -> MaxPool
    """
    inputs = layers.Input(shape=input_shape)

    # 1x1 Conv
    x = layers.Conv1D(128, 1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)

    # 1x3 Conv
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)

    # 1x7 Conv (Stride 2 implied by diagram's next stage or specific implementation, 
    # but diagram shows MaxPool doing the reduction. We stick to diagram: Conv -> Pool)
    x = layers.Conv1D(128, 7, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)

    # Max Pooling 1x3, stride 2 (matches "Max Pooling: 1x3, /2")
    outputs = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    return models.Model(inputs, outputs, name="ProjectionHead")

def build_encoder(input_shape_projected=(32, 128), latent_dim=256):
    """
    Matches "Encoder" block in Fig. 5:
    - Stage 1 (Blue): 3 blocks, 64 filters (expands to 256)
    - Stage 2 (Yellow): 3 blocks, 128 filters (expands to 512)
    - Global Average Pooling -> Linear
    """
    inputs = layers.Input(shape=input_shape_projected)
    x = inputs

    # --- Stage 1 (Blue) ---
    # First block downsamples (stride=2) if needed, but diagram shows "/2" on the shortcut.
    # We follow standard ResNet stage logic: First block strides, others don't.
    x = bottleneck_block(filters=64, stride=2, expansion=4)(x) # Output: (16, 256)
    x = bottleneck_block(filters=64, stride=1, expansion=4)(x)
    x = bottleneck_block(filters=64, stride=1, expansion=4)(x)

    # --- Stage 2 (Yellow) ---
    x = bottleneck_block(filters=128, stride=2, expansion=4)(x) # Output: (8, 512)
    x = bottleneck_block(filters=128, stride=1, expansion=4)(x)
    x = bottleneck_block(filters=128, stride=1, expansion=4)(x)

    # --- Output Head ---
    x = layers.GlobalAveragePooling1D()(x) # (512,)
    z = layers.Dense(latent_dim, name="z_latent")(x) # (256,)

    return models.Model(inputs, z, name="Encoder")

def build_decoder(latent_dim=256, output_steps=64, output_features=38):
    """
    Matches "Decoder" block in Fig. 5 (Symmetric to Encoder):
    - Linear -> Reshape
    - Stage 2 Inverse (Yellow): 3 blocks
    - Stage 1 Inverse (Blue): 3 blocks
    - Final Conv Projection
    """
    inputs = layers.Input(shape=(latent_dim,))

    # Linear Projection & Reshape (Inverse of Global Avg Pool)
    # Target shape: (8, 512) to match Encoder's Stage 2 output
    x = layers.Dense(8 * 512)(inputs)
    x = layers.Reshape((8, 512))(x)
    
    # --- Stage 2 Inverse (Yellow) ---
    x = bottleneck_block(filters=128, stride=1, expansion=4, is_decoder=True)(x)
    x = bottleneck_block(filters=128, stride=1, expansion=4, is_decoder=True)(x)
    x = bottleneck_block(filters=128, stride=2, expansion=4, is_decoder=True)(x) # Upsample to (16, 512)

    # --- Stage 1 Inverse (Blue) ---
    # Note: Encoder Stage 1 output was (16, 256). 
    # We need to transition from 512 -> 256 filters here.
    # The bottleneck block takes 'filters' (e.g. 64) and expands to 256.
    x = bottleneck_block(filters=64, stride=1, expansion=4, is_decoder=True)(x)
    x = bottleneck_block(filters=64, stride=1, expansion=4, is_decoder=True)(x)
    x = bottleneck_block(filters=64, stride=2, expansion=4, is_decoder=True)(x) # Upsample to (32, 256)

    # --- Final Projection Head (Inverse of Input Projection) ---
    # Matches "Conv: 1x1, 128" -> "Linear" (Output)
    # We upsample to original time length (64)
    x = layers.UpSampling1D(size=2)(x) # (64, 256)
    
    x = layers.Conv1D(128, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)

    # Final Linear Output (Conv 1x1 to feature count)
    outputs = layers.Conv1D(output_features, 1, padding='same', activation=None)(x)

    return models.Model(inputs, outputs, name="Decoder")

def build_discriminator(latent_dim=256):
    """
    Matches MLP Discriminator for feature decomposition.
    Input: Concatenated [Original Z, Composite Z] (Size: 512)
    Output: [Class Label, Proportion Alpha] (Size: 2)
    Activation: Linear (None) for MSE Loss
    """
    input_dim = latent_dim * 2
    inputs = layers.Input(shape=(input_dim,))

    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)

    # No Sigmoid! Paper uses MSE loss on raw values.
    outputs = layers.Dense(2, activation=None)(x) 
    return models.Model(inputs, outputs, name="Discriminator")

