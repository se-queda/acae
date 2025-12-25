import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# Use namespace as per your preference
namespace = tf.keras

class ResidualBlock(layers.Layer):
    """
    Stabilizes gradient flow for the Hamiltonian network.
    Mirroring the ResNet-1D architecture from the paper[cite: 387].
    """
    def __init__(self, units, dropout=0.1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.units = units
        self.dense1 = layers.Dense(units, activation='tanh', 
                                   kernel_regularizer=regularizers.l2(1e-4))
        self.bn = layers.BatchNormalization()
        self.dropout = layers.Dropout(dropout)
        self.dense2 = layers.Dense(units, activation='tanh', 
                                   kernel_regularizer=regularizers.l2(1e-4))
        self.shortcut_layer = None

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.shortcut_layer = layers.Dense(self.units)
        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs):
        shortcut = inputs
        x = self.dense1(inputs)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.dense2(x)
        if self.shortcut_layer:
            shortcut = self.shortcut_layer(shortcut)
        # FIX: Use raw tensor addition instead of instantiating layers.Add() in call
        return x + shortcut

class SymplecticLeapfrogLayer(layers.Layer):
    """
    The Physics Engine. Replaces the 1D-CNN backbone.
    dq/dt = dH/dp, dp/dt = -dH/dq.
    """
    def __init__(self, feature_dim=128, steps=3, dt=0.1, **kwargs):
        super(SymplecticLeapfrogLayer, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.steps = steps
        self.dt = dt
        
        # Scalar Energy Function H(q, p)
        # We use a pure MLP here to ensure a smooth second-order gradient path.
        self.h_dense1 = layers.Dense(256, activation='tanh')
        self.h_dense2 = layers.Dense(256, activation='tanh')
        # FIX: use_bias=False prevents the "No gradients exist" warning for bias
        self.h_out = layers.Dense(1, use_bias=False, name="h_out") 

    def get_gradients(self, q, p):
        state_p = tf.concat([q, p], axis=-1)
        with tf.GradientTape() as tape:
            tape.watch(state_p)
            x = self.h_dense1(state_p)
            x = self.h_dense2(x)
            H = self.h_out(x)
        dH = tape.gradient(H, state_p)
        return dH[:, self.feature_dim:], -dH[:, :self.feature_dim]

    def call(self, x):
        # Extract initial canonical coordinates
        q = x[:, -1, :] 
        p = x[:, -1, :] - x[:, -2, :] 
        
        for _ in range(self.steps):
            dq, dp = self.get_gradients(q, p)
            p = p + 0.5 * self.dt * dp
            q = q + self.dt * dq
            _, dp_final = self.get_gradients(q, p)
            p = p + 0.5 * self.dt * dp_final
            
        return tf.concat([q, p], axis=-1)

def build_projection_head(input_shape=(64, 22)):
    # Matches the paper's multi-scale architecture [cite: 338, 380]
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(128, 1, padding='same', activation='tanh')(inputs)
    x = layers.Conv1D(128, 3, padding='same', activation='tanh')(x)
    x = layers.Conv1D(128, 7, padding='same', activation='tanh')(x)
    return models.Model(inputs, x, name="ProjectionHead")

def build_dual_encoder(input_shape_sys, input_shape_res, latent_dim=256):
    # Inputs for both anchors
    inputs_sys = layers.Input(shape=input_shape_sys, name="input_sys")
    inputs_res = layers.Input(shape=input_shape_res, name="input_res")
    
    # --- Branch A: System Dynamics (HNN) ---
    proj_sys = build_projection_head(input_shape_sys)(inputs_sys)
    flow = SymplecticLeapfrogLayer(feature_dim=128)(proj_sys)
    x_sys = ResidualBlock(512)(flow)
    z_sys = layers.Dense(latent_dim // 2, name="z_sys")(x_sys)
    
    # --- Branch B: Residual Sentinel (Identity) ---
    # We use a simpler path for the lone-wolf/dead sensors 
    # to avoid polluting the HNN manifold.
    x_res = layers.Flatten()(inputs_res)
    x_res = layers.Dense(256, activation='tanh')(x_res)
    z_res = layers.Dense(latent_dim // 2, name="z_res")(x_res)
    
    # Concatenate for the Discriminator, but keep them separate for Decoders
    z_combined = layers.Concatenate(name="z_combined")([z_sys, z_res])
    
    return models.Model([inputs_sys, inputs_res], [z_sys, z_res, z_combined], name="DualEncoder")

def build_dual_decoder(latent_dim=256, output_steps=64, feat_sys=22, feat_res=16):
    # We use half the latent for each branch
    z_sys_in = layers.Input(shape=(latent_dim // 2,))
    z_res_in = layers.Input(shape=(latent_dim // 2,))
    
    # --- Decoder A: Reconstruct System Dynamics ---
    x_s = layers.Dense(512, activation='tanh')(z_sys_in)
    x_s = ResidualBlock(512)(x_s)
    x_s = layers.Dense(output_steps * 128, activation='tanh')(x_s)
    x_s = layers.Reshape((output_steps, 128))(x_s)
    out_sys = layers.Conv1D(feat_sys, 1, padding='same')(x_s)
    
    # --- Decoder B: Reconstruct Residual/Sentinel ---
    # This branch learns the "Silence" (0.0) of dead features 
    # and the "Noise" of lone wolves.
    x_r = layers.Dense(256, activation='tanh')(z_res_in)
    x_r = layers.Dense(output_steps * feat_res, activation='tanh')(x_r)
    out_res = layers.Reshape((output_steps, feat_res))(x_r)
    
    return models.Model([z_sys_in, z_res_in], [out_sys, out_res], name="DualDecoder")

def build_discriminator(latent_dim=256):
    # Standard ACAE decomposition MLP [cite: 430, 436]
    inputs = layers.Input(shape=(latent_dim * 2,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(2, activation=None)(x) 
    return models.Model(inputs, outputs, name="Discriminator")