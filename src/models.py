import tensorflow as tf
from tensorflow.keras import layers, models

def residual_block(filters, downsample=False):
    def block(x_input):
        stride = 2 if downsample else 1
        x = layers.Conv1D(filters, 3, padding='same', strides=stride, activation='relu')(x_input)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters, 3, padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)
        if downsample or x_input.shape[-1] != filters:
            x_input = layers.Conv1D(filters, 1, strides=stride, padding='same')(x_input)
        out = layers.Add()([x, x_input])
        return layers.Activation('relu')(out)
    return block

# --- 1. PROJECTION HEAD (New) ---
def build_projection_head(input_shape=(64, 38), projection_dim=256):
    """
    Projects raw features (38) to latent space (256) per timestep.
    This creates the space where masking "zeros" are distinct from data "zeros".
    """
    inputs = layers.Input(shape=input_shape)
    # Dense applied to last dimension = Linear Projection per timestep
    outputs = layers.Dense(projection_dim, activation=None, name="Projection_Linear")(inputs)
    return models.Model(inputs, outputs, name="ProjectionHead")

# --- 2. ENCODER (Updated with Stem) ---
def build_encoder(input_shape=(64, 256), latent_dim=256):
    inputs = layers.Input(shape=input_shape)

    # Stem (Standard ResNet-1D start)
    x = layers.Conv1D(64, 7, strides=2, padding='same', name="Stem_Conv")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same', name="Stem_Pool")(x)
    # Shape is now roughly (16, 64) due to /2 stride and /2 pool

    # Residual Blocks
    x = residual_block(64)(x)                   # (16, 64)
    x = residual_block(128, downsample=True)(x) # (8, 128)
    x = residual_block(128)(x)                  # (8, 128)
    x = residual_block(256, downsample=True)(x) # (4, 256)

    x = layers.GlobalAveragePooling1D()(x)      # (256,)
    z = layers.Dense(latent_dim, name="Z_Representation")(x) 
    return models.Model(inputs, z, name="Encoder")

# --- 3. DECODER (Updated for Symmetry) ---
def build_decoder(latent_dim=256, output_shape=(64, 38)):
    time_steps, features = output_shape
    inputs = layers.Input(shape=(latent_dim,))

    # Start from compressed representation (inverse of GlobalAvg + last block)
    x = layers.Dense(4 * 256)(inputs) 
    x = layers.Reshape((4, 256))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Symmetric Upsampling (Conv1DTranspose)
    # Block 3 Inverse (4 -> 8)
    x = layers.Conv1DTranspose(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Block 2 Inverse (8 -> 16)
    x = layers.Conv1DTranspose(64, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Stem Inverse (16 -> 32 -> 64)
    x = layers.Conv1DTranspose(64, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv1DTranspose(32, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Final reconstruction to raw feature count
    outputs = layers.Conv1D(features, 3, padding='same', activation=None)(x)
    return models.Model(inputs, outputs, name="Decoder")

# --- 4. DISCRIMINATOR (Updated for Pair Input) ---
def build_discriminator(latent_dim=256):
    # Inputs: [Reference Z, Composite Z] -> Size is 2 * latent_dim
    input_dim = latent_dim * 2
    inputs = layers.Input(shape=(input_dim,))

    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)

    outputs = layers.Dense(2, activation='sigmoid')(x)  # [label, proportion]
    return models.Model(inputs, outputs, name="Discriminator")