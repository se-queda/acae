import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from src.mamba import ResidualBlock, ModelArgs

# Preferred persona namespace
namespace = tf.keras

REG = 1e-4


class MambaProjector(layers.Layer):
    """
    Projects raw telemetry signals into Mamba channel space.

    Input:  (B, L, F)   raw sensors
    Output: (B, L, D)   Mamba channels
    """

    def __init__(self, out_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim

        # Feature mixing (sensor → latent channels)
        self.feature_proj = layers.Dense(
            out_dim,
            use_bias=True
        )

        # Light temporal smoothing (important for telemetry)
        self.temporal_conv = layers.Conv1D(
            filters=out_dim,
            kernel_size=5,
            padding="same",
            groups=1,
            use_bias=True
        )

        self.norm = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x):
        """
        x: (B, L, F)
        """
        # Mix sensors into channel space
        x = self.feature_proj(x)          # (B, L, D)

        # Smooth short-term noise (before SSM)
        x = self.temporal_conv(x)          # (B, L, D)

        # Normalize for SSM stability
        x = self.norm(x)

        return x

def mamba_encoder(inputs, args):
    # inputs: (B, L, F)

    x = MambaProjector(out_dim=args.model_input_dims)(inputs)

    for _ in range(2):
        x = ResidualBlock(args)(x)

    latent = layers.GlobalAveragePooling1D()(x)
    return latent




def fno_lifting(x, latent_dim=256):
    x = layers.Dense(latent_dim, 
                     kernel_regularizer=regularizers.l2(REG),
                     name="FNO_Lifting")(x)
    # Jerk masking happens here or on the raw input as you requested
    x = layers.Activation('gelu')(x) 
    return x
class SpectralConv1D(layers.Layer):
    def __init__(self, out_channels, modes, **kwargs):
        super(SpectralConv1D, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.modes = modes

    def build(self, input_shape):
        in_channels = input_shape[-1]
        # Ensure weights are initialized with fully-defined shapes
        self.weights_real = self.add_weight(
            shape=(self.modes, in_channels, self.out_channels),
            initializer='glorot_normal',
            trainable=True,
            name='spectral_weights_real'
        )
        self.weights_imag = self.add_weight(
            shape=(self.modes, in_channels, self.out_channels),
            initializer='zeros',
            trainable=True,
            name='spectral_weights_imag'
        )

    def call(self, x):
        # x shape: (Batch, Time, Channels)
        n = tf.shape(x)[1]
        
        # 1. Fourier Transform
        x = tf.transpose(x, perm=[0, 2, 1]) 
        x_ft = tf.signal.rfft(x) 
        
        # 2. Spectral Filter
        x_ft_low = x_ft[:, :, :self.modes]
        weights = tf.complex(self.weights_real, self.weights_imag)
        out_ft_low = tf.einsum('bim,mio->bom', x_ft_low, weights)
        
        # 3. Inverse Transform & Resolution Recovery
        padding = tf.zeros([tf.shape(out_ft_low)[0], self.out_channels, (n // 2 + 1) - self.modes], dtype=tf.complex64)
        out_ft = tf.concat([out_ft_low, padding], axis=-1)
        
        x = tf.signal.irfft(out_ft, fft_length=[n])
        x = tf.transpose(x, perm=[0, 2, 1])
        
        # --- THE FIX ---
        # Explicitly restore the static channel dimension for BatchNormalization
        # This prevents the 'None' dimension error during variable initialization.
        x.set_shape([None, None, self.out_channels])
        return x

    def compute_output_shape(self, input_shape):
        """Mandatory for custom layers to help Keras trace shapes."""
        return (input_shape[0], input_shape[1], self.out_channels)

def fno_block(x, filters, modes, dropout=0.1):
    """NVIDIA Physics-ML Block: Fourier Path + Pointwise Shortcut."""
    shortcut = namespace.layers.Conv1D(filters, 1, padding='same', 
                                      kernel_regularizer=namespace.regularizers.l2(REG))(x)
    
    # This call now carries the static shape info correctly
    x_f = SpectralConv1D(filters, modes)(x)
    x_f = namespace.layers.BatchNormalization()(x_f)
    
    x = namespace.layers.Add()([x_f, shortcut])
    x = namespace.layers.Activation('gelu')(x)
    x = namespace.layers.SpatialDropout1D(dropout)(x)
    return x


REG = 1e-4


def build_dual_encoder(input_shape_sys, input_shape_res, config):
    """
    Dual-Anchor Encoder
    - Branch A: Mamba-based consensus encoder (replaces HNN)
    - Branch B: FNO residual encoder
    """

    # --------------------------------------------------
    # Config
    # --------------------------------------------------
    L = config.get("latent_dim", 512)              # total latent
    dim = config.get("mamba_dim", 256)           # mamba channel dim
    drop_rate = config.get("dropout", 0.0)         # keep 0 for now
    f_modes = config.get("fno_modes", 32)

    feat_sys = input_shape_sys[1]

    # --------------------------------------------------
    # Inputs
    # --------------------------------------------------
    inputs_sys = layers.Input(shape=input_shape_sys, name="input_sys")
    inputs_res = layers.Input(shape=input_shape_res, name="input_res")

    # ==================================================
    # Branch A: Mamba Consensus Encoder (replaces HNN)
    # ==================================================
    if feat_sys > 0:
        mamba_args = ModelArgs(
            model_input_dims=dim,
            model_states=config.get("mamba_states", 32),
            seq_length=input_shape_sys[0]
        )

        sys_latent = mamba_encoder(inputs_sys, mamba_args)  # (B, h_dim)

        z_sys = layers.Dense(
            L // 2,
            name="z_sys",
            kernel_regularizer=regularizers.l2(REG)
        )(sys_latent)

    else:
        z_sys = layers.Lambda(
            lambda x: tf.zeros((tf.shape(x)[0], L // 2)),
            name="z_sys"
        )(inputs_sys)


    # ==================================================
    # Branch B: FNO Residual Encoder (unchanged)
    # ==================================================
    x_fno = fno_lifting(inputs_res, dim)

    for _ in range(config.get("fno_blocks", 4)):
        x_fno = fno_block(x_fno, dim, modes=f_modes)

    x_fno = layers.GlobalAveragePooling1D()(x_fno)

    z_res = layers.Dense(
        L // 2,
        name="z_res",
        kernel_regularizer=regularizers.l2(REG)
    )(x_fno)

    # ==================================================
    # Joint Latent
    # ==================================================
    z_combined = layers.Concatenate(name="z_combined")([z_sys, z_res])

    return models.Model(
        inputs=[inputs_sys, inputs_res],
        outputs=[z_sys, z_res, z_combined],
        name="DualAnchor_MambaFNO_Encoder"
    )



def build_dual_decoder(feat_sys, feat_res, output_steps, config):
    L = config.get("latent_dim", 512)
    dim = config.get("mamba_dim", 256)
    f_modes = config.get("fno_modes", 8)
    drop_rate = config.get("dropout", 0.1)

    # --------------------------------------------------
    # Inputs
    # --------------------------------------------------
    z_sys_in = layers.Input(shape=(L // 2,), name="z_sys_input")
    z_res_in = layers.Input(shape=(L // 2,), name="z_res_input")

    # ==================================================
    # Decoder A: Consensus / Physics Reconstruction
    # ==================================================
    x_s = layers.Dense(
        dim,
        activation="gelu",
        kernel_regularizer=regularizers.l2(REG)
    )(z_sys_in)

    x_s = layers.Dense(
        dim,
        activation="gelu",
        kernel_regularizer=regularizers.l2(REG)
    )(x_s)

    x_s = layers.Dense(
        output_steps * dim,
        activation="gelu",
        kernel_regularizer=regularizers.l2(REG)
    )(x_s)

    x_s = layers.Reshape((output_steps, dim))(x_s)

    if feat_sys > 0:
        out_sys = layers.Conv1D(
            feat_sys,
            kernel_size=1,
            padding="same",
            activation="linear",
            kernel_regularizer=regularizers.l2(REG),
            name="out_phy"
        )(x_s)
    else:
        out_sys = layers.Lambda(
            lambda x: tf.zeros((tf.shape(x)[0], output_steps, 0)),
            name="out_phy"
        )(x_s)

    # ==================================================
    # Decoder B: Residual / FNO Reconstruction
    # ==================================================
    x_r = layers.Dense(
        dim,
        activation="gelu",
        kernel_regularizer=regularizers.l2(REG)
    )(z_res_in)

    x_r = layers.Dense(
        output_steps * dim,
        activation="gelu",
        kernel_regularizer=regularizers.l2(REG)
    )(x_r)

    x_r = layers.Reshape((output_steps, dim))(x_r)

    for _ in range(config.get("fno_decoder_blocks", 2)):
        x_r = fno_block(x_r, dim, modes=f_modes, dropout=drop_rate)

    x_r = layers.Conv1D(
        feat_res,
        kernel_size=1,
        padding="same",
        kernel_regularizer=regularizers.l2(REG)
    )(x_r)

    out_res = layers.Activation("linear", name="out_res")(x_r)

    return models.Model(
        inputs=[z_sys_in, z_res_in],
        outputs=[out_sys, out_res],
        name="DualAnchor_MambaFNO_Decoder"
    )



def build_discriminator(input_dim):
    # ACAE Discriminator uses dropout of 0.5 
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dropout(0.5)(x) 
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation=None)(x) 
    return models.Model(inputs, outputs)