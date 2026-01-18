import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from einops import rearrange
from dataclasses import dataclass


# ============================================================
# Model Arguments (fixed defaults, no tuning for now)
# ============================================================
@dataclass
class ModelArgs:
    model_input_dims: int = 256     # latent channel size (projected sensors)
    model_states: int = 16          # SSM hidden states
    projection_expand_factor: int = 2
    conv_kernel_size: int = 8        # slightly larger receptive field
    delta_t_min: float = 1e-4
    delta_t_max: float = 0.1
    seq_length: int = 1000           # SMD / PSM window length


# ============================================================
# Selective Scan (SSM core)
# ============================================================
def selective_scan(u, delta, A, B, C, D):
    """
    u:     (B, L, D)
    delta: (B, L, D)
    A:     (D, N)
    B,C:   (B, L, N)
    D:     (D,)
    """

    # ΔA term
    dA = tf.einsum('bld,dn->bldn', delta, A)

    # ΔBu term
    dB_u = tf.einsum('bld,bld,bln->bldn', delta, u, B)

    # Prefix-scan trick (parallel cumulative sum)
    dA_cumsum = tf.pad(dA[:, 1:], [[0, 0], [1, 1], [0, 0], [0, 0]])[:, 1:]
    dA_cumsum = tf.reverse(dA_cumsum, axis=[1])
    dA_cumsum = tf.math.cumsum(dA_cumsum, axis=1)
    dA_cumsum = tf.exp(dA_cumsum)
    dA_cumsum = tf.reverse(dA_cumsum, axis=[1])

    # State evolution
    x = dB_u * dA_cumsum
    x = tf.math.cumsum(x, axis=1) / (dA_cumsum + 1e-12)

    # Output projection
    y = tf.einsum('bldn,bln->bld', x, C)

    return y + u * D


# ============================================================
# Mamba Block
# ============================================================
class MambaBlock(layers.Layer):
    def __init__(self, args: ModelArgs, **kwargs):
        super().__init__(**kwargs)
        self.args = args

        D = args.model_input_dims
        N = args.model_states
        delta_rank = D // 16

        # Input projection → (x, residual)
        self.in_projection = layers.Dense(2 * D, use_bias=False)

        # Depthwise causal convolution (local smoothing)
        self.conv1d = layers.Conv1D(
            filters=D,
            kernel_size=args.conv_kernel_size,
            groups=D,
            padding="causal",
            data_format="channels_first",
            use_bias=True,
        )

        # Projection to Δ, B, C
        self.x_projection = layers.Dense(delta_rank + 2 * N, use_bias=False)

        # Δ projection to full channel dimension
        self.delta_projection = layers.Dense(D, use_bias=True)

        # --------------------------------------------------------
        # CRITICAL FIX:
        # A_log must be (D, N), NOT (N,)
        # Each channel has its own decay spectrum
        # --------------------------------------------------------
        A_init = tf.range(1, N + 1, dtype=tf.float32)
        A_init = tf.repeat(A_init[None, :], D, axis=0)
        self.A_log = tf.Variable(tf.math.log(A_init), trainable=True)

        # Skip connection coefficient
        self.D = tf.Variable(tf.ones(D), trainable=True)

        # Output projection
        self.out_projection = layers.Dense(D, use_bias=False)

    def call(self, x):
        """
        x: (B, L, D)
        """
        B, L, _ = x.shape

        # Split input and residual
        x_and_res = self.in_projection(x)
        x, res = tf.split(x_and_res, 2, axis=-1)

        # Depthwise causal convolution
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :L]
        x = rearrange(x, 'b d l -> b l d')

        x = tf.nn.swish(x)

        # State-space model
        y = self.ssm(x)

        # Gated residual
        y = y * tf.nn.swish(res)

        return self.out_projection(y)

    def ssm(self, x):
        """
        Runs the selective SSM
        """
        D = self.args.model_input_dims
        N = self.args.model_states
        delta_rank = D // 16

        # Stable negative eigenvalues
        A = -tf.exp(self.A_log)

        # Input-dependent parameters
        x_proj = self.x_projection(x)
        delta, B, C = tf.split(
            x_proj,
            [delta_rank, N, N],
            axis=-1
        )

        # --------------------------------------------------------
        # CRITICAL FIX:
        # Clamp Δ to avoid exploding / vanishing dynamics
        # --------------------------------------------------------
        delta = tf.nn.softplus(self.delta_projection(delta))
        delta = tf.clip_by_value(
            delta,
            self.args.delta_t_min,
            self.args.delta_t_max
        )

        return selective_scan(x, delta, A, B, C, self.D)


# ============================================================
# Residual Block (Pre-Norm, Stable)
# ============================================================
class ResidualBlock(layers.Layer):
    def __init__(self, args: ModelArgs, **kwargs):
        super().__init__(**kwargs)
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.mixer = MambaBlock(args)

    def call(self, x):
        return self.mixer(self.norm(x)) + x

