import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# Use namespace as per your preference
namespace = tf.keras

def residual_mlp(units, dropout=0.1):
    """
    Stabilizes gradient flow for the Hamiltonian network.
    Uses L2 regularization to prevent overfitting[cite: 100].
    """
    def block(x_input):
        shortcut = x_input
        x = layers.Dense(units, activation='tanh', 
                         kernel_regularizer=regularizers.l2(1e-4))(x_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(units, activation='tanh', 
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        
        # Projection shortcut if dimensions don't match
        if shortcut.shape[-1] != units:
            shortcut = layers.Dense(units)(shortcut)
            
        return layers.Add()([x, shortcut])
    return block

class SymplecticLeapfrogLayer(layers.Layer):
    """
    The Physics Engine. Replaces the 1D-CNN backbone[cite: 386].
    Simulates Hamiltonian dynamics: dq/dt = dH/dp, dp/dt = -dH/dq.
    """
    def __init__(self, feature_dim=128, steps=3, dt=0.1, **kwargs):
        super(SymplecticLeapfrogLayer, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.steps = steps
        self.dt = dt
        
        # Scalar Energy Function H(q, p)
        self.h_net = models.Sequential([
            residual_mlp(256),
            residual_mlp(256),
            layers.Dense(1) 
        ])

    def get_gradients(self, q, p):
        state_p = tf.concat([q, p], axis=-1)
        with tf.GradientTape() as tape:
            tape.watch(state_p)
            H = self.h_net(state_p)
        dH = tape.gradient(H, state_p)
        # Standard Hamiltonian equations
        return dH[:, self.feature_dim:], -dH[:, :self.feature_dim]

    def call(self, x):
        # Extract State (q) and Internal Jerk (p) from the projected window
        q = x[:, -1, :] 
        p = x[:, -1, :] - x[:, -2, :] 
        
        # 3-step Symplectic Integration for physical stability
        for _ in range(self.steps):
            dq, dp = self.get_gradients(q, p)
            p = p + 0.5 * self.dt * dp
            q = q + self.dt * dq
            _, dp_final = self.get_gradients(q, p)
            p = p + 0.5 * self.dt * dp_final
            
        return tf.concat([q, p], axis=-1)

def build_projection_head(input_shape=(64, 22)):
    """
    Matches Fig. 5 Projection Head: 1x1, 1x3, 1x7 Convs[cite: 338, 380].
    Handles the high-dimensional projection for masking[cite: 289].
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(128, 1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)
    
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)
    
    x = layers.Conv1D(128, 7, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)
    
    # Reduction step as per paper diagram [cite: 345, 380]
    outputs = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    return models.Model(inputs, outputs, name="ProjectionHead")

def build_encoder(input_shape=(64, 22), latent_dim=256):
    """
    ACAE Encoder with Hamiltonian backbone[cite: 95, 332].
    Outputs the standard 256-dim latent vector z[cite: 384, 544].
    """
    inputs = layers.Input(shape=input_shape)
    
    # 1. Multi-scale Projection [cite: 338]
    projected = build_projection_head(input_shape)(inputs)
    
    # 2. Physics Engine (HNN)
    flow = SymplecticLeapfrogLayer(feature_dim=128)(projected)
    
    # 3. Final Latent Mapping [cite: 544]
    x = residual_mlp(512)(flow)
    z = layers.Dense(latent_dim, name="z_latent")(x)
    
    return models.Model(inputs, z, name="Encoder")

def build_decoder(latent_dim=256, output_steps=64, output_features=22):
    """
    Symmetric Decoder: Mirror image of the Encoder.
    Reconstructs original sensor window from physical latent flow[cite: 467, 468].
    """
    z_input = layers.Input(shape=(latent_dim,))
    
    x = residual_mlp(512)(z_input)
    x = layers.Dense(256, activation='tanh')(x) 
    
    # Expand to time window [cite: 275, 466]
    x = layers.Dense(output_steps * 128, activation='tanh')(x)
    x = layers.Reshape((output_steps, 128))(x)
    
    # Inverse Projection back to sensor count
    x = layers.Conv1D(64, 3, padding='same', activation='tanh')(x)
    outputs = layers.Conv1D(output_features, 1, padding='same')(x)
    
    return models.Model(z_input, outputs, name="Decoder")

def build_discriminator(latent_dim=256):
    """
    Standard ACAE MLP Discriminator[cite: 430].
    Used for adversarial feature decomposition[cite: 395, 435].
    """
    input_dim = latent_dim * 2
    inputs = layers.Input(shape=(input_dim,))

    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)

    # Output: [Classification Label, Proportion Alpha] [cite: 430, 439]
    outputs = layers.Dense(2, activation=None)(x) 
    return models.Model(inputs, outputs, name="Discriminator")
