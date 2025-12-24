import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

namespace = tf.keras

class ResidualBlock(layers.Layer):
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
        return layers.Add()([x, shortcut])

class SymplecticLeapfrogLayer(layers.Layer):
    """
    The Physics Engine. Explicitly tracks sub-model weights for the optimizer.
    """
    def __init__(self, feature_dim=128, steps=3, dt=0.1, **kwargs):
        super(SymplecticLeapfrogLayer, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.steps = steps
        self.dt = dt
        
        # Define H(q, p) with an explicit InputLayer to finalize weights [cite: 387, 544]
        self.h_net = models.Sequential([
            layers.Input(shape=(feature_dim * 2,)), 
            ResidualBlock(256),
            ResidualBlock(256),
            layers.Dense(1) 
        ])

    def build(self, input_shape):
        # Force-build the sub-model so dense_4/bias exists before training 
        if not self.h_net.built:
            self.h_net.build((None, self.feature_dim * 2))
        super(SymplecticLeapfrogLayer, self).build(input_shape)

    @property
    def trainable_variables(self):
        # EXPLICIT REGISTRATION: Bubbles internal weights up to the Encoder 
        return super(SymplecticLeapfrogLayer, self).trainable_variables + self.h_net.trainable_variables

    def get_gradients(self, q, p):
        state_p = tf.concat([q, p], axis=-1)
        with tf.GradientTape() as tape:
            tape.watch(state_p)
            H = self.h_net(state_p)
        dH = tape.gradient(H, state_p)
        # Physics flow: dq/dt = dH/dp, dp/dt = -dH/dq
        return dH[:, self.feature_dim:], -dH[:, :self.feature_dim]

    def call(self, x):
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
    # Mirroring the paper's multi-scale conv architecture [cite: 338, 380]
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(128, 1, padding='same', activation='tanh')(inputs)
    x = layers.Conv1D(128, 3, padding='same', activation='tanh')(x)
    x = layers.Conv1D(128, 7, padding='same', activation='tanh')(x)
    return models.Model(inputs, x, name="ProjectionHead")

def build_encoder(input_shape=(64, 22), latent_dim=256):
    inputs = layers.Input(shape=input_shape)
    projected = build_projection_head(input_shape)(inputs)
    flow = SymplecticLeapfrogLayer(feature_dim=128)(projected)
    x = ResidualBlock(512)(flow)
    z = layers.Dense(latent_dim, name="z_latent")(x)
    return models.Model(inputs, z, name="Encoder")

def build_decoder(latent_dim=256, output_steps=64, output_features=22):
    # Precise inverse of the encoder to facilitate robust reconstruction [cite: 467, 472]
    z_input = layers.Input(shape=(latent_dim,))
    x = layers.Dense(512, activation='tanh')(z_input)
    x = ResidualBlock(512)(x)
    x = layers.Dense(output_steps * 128, activation='tanh')(x)
    x = layers.Reshape((output_steps, 128))(x)
    x = layers.Conv1D(128, 7, padding='same', activation='tanh')(x)
    x = layers.Conv1D(128, 3, padding='same', activation='tanh')(x)
    outputs = layers.Conv1D(output_features, 1, padding='same')(x)
    return models.Model(z_input, outputs, name="Decoder")

def build_discriminator(latent_dim=256):
    # Standard ACAE decomposition MLP [cite: 430, 436]
    inputs = layers.Input(shape=(latent_dim * 2,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(2, activation=None)(x) 
    return models.Model(inputs, outputs, name="Discriminator")
