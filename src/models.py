import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# Preferred persona namespace
namespace = tf.keras

REG = 1e-4

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
        return x + shortcut
    
    
def tcn_projector(x,projection_dim = 256):
    x = layers.Dense(projection_dim, 
                     kernel_regularizer=regularizers.l2(REG),
                     name="Linear_Projection")(x)
    # The paper applies the mask AFTER this projection[cite: 296, 307].
    return x

def hnn_projector(x, projection_dim=256):
    # 1x1: Point-wise semantic mapping
    x = layers.Conv1D(projection_dim, 1, padding='same', activation='tanh',
                      kernel_regularizer=regularizers.l2(REG))(x)
    # 3x3: Local neighborhood trends
    x = layers.Conv1D(projection_dim, 3, padding='same', activation='tanh',
                      kernel_regularizer=regularizers.l2(REG))(x)
    # 7x7: Wider contextual window
    x = layers.Conv1D(projection_dim, 7, padding='same', activation='tanh',
                      kernel_regularizer=regularizers.l2(REG))(x)
    x = layers.BatchNormalization()(x)
    return x

class SymplecticLeapfrogLayer(layers.Layer):
    def __init__(self, feature_dim, steps, dt, **kwargs):
        super(SymplecticLeapfrogLayer, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.steps = steps
        self.dt = dt
        self.h_dense1 = layers.Dense(feature_dim * 2, activation='relu')
        self.h_dense2 = layers.Dense(feature_dim * 2, activation='relu')
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
        q = x[:, -1, :] 
        p = x[:, -1, :] - x[:, -2, :] 
        for _ in range(self.steps):
            dq, dp = self.get_gradients(q, p)
            p = p + 0.5 * self.dt * dp
            q = q + self.dt * dq
            _, dp_final = self.get_gradients(q, p)
            p = p + 0.5 * self.dt * dp_final
        return tf.concat([q, p], axis=-1)



# Global Regularization Factor
REG = 1e-4

def tcn_block(x, filters, dilation, dropout=0.1):
    """
    Standard Residual TCN Block with L2 Regularization and Spatial Dropout.
    """
    shortcut = x
    
    # Adjust shortcut dimension if filters don't match
    if x.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same', 
                                 kernel_regularizer=regularizers.l2(REG))(x)
        
    # Dilated Convolution Path
    x = layers.Conv1D(filters, kernel_size=3, dilation_rate=dilation, 
                      padding='same', kernel_regularizer=regularizers.l2(REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout1D(dropout)(x) # Better than standard dropout for temporal data
    
    x = layers.Conv1D(filters, kernel_size=3, dilation_rate=dilation, 
                      padding='same', kernel_regularizer=regularizers.l2(REG))(x)
    x = layers.BatchNormalization()(x)
    
    # Residual Addition
    x = layers.Add()([x, shortcut])
    return layers.Activation('relu')(x)


def build_dual_encoder(input_shape_sys, input_shape_res, config):
    L = config.get("latent_dim", 512) 
    h_dim = config.get("hnn_feature_dim", 256) 
    drop_rate = config.get("dropout", 0.1)
    feat_sys = input_shape_sys[1]
    inputs_sys = layers.Input(shape=input_shape_sys, name="input_sys")
    inputs_res = layers.Input(shape=input_shape_res, name="input_res")
    
    if feat_sys>0:
        # --- Branch A: Hamiltonian Dynamics (Consensus) ---
        proj_hnn = hnn_projector(inputs_sys, projection_dim=h_dim)
        
        # Leapfrog remains the core physics engine
        flow = SymplecticLeapfrogLayer(
            feature_dim=h_dim, 
            steps=config.get("hnn_steps", 3), 
            dt=config.get("hnn_dt", 0.1)
        )(proj_hnn)
        
        # Added Residual Connection around HNN output
        x_sys = layers.Dense(L // 2, activation='tanh', 
                                kernel_regularizer=regularizers.l2(REG))(flow)
        x_sys = layers.Dropout(drop_rate)(x_sys)
        z_sys = layers.Dense(L // 2, name="z_sys")(x_sys)
    else:
        z_sys = layers.Lambda(lambda x: tf.zeros((tf.shape(x)[0], L // 2)), name="z_sys")(inputs_sys)
    
    # --- Branch B: TCN (Uses Paper's Faithful Linear Projection) ---
    x_tcn = tcn_projector(inputs_res, h_dim) 
    x_tcn = tcn_block(x_tcn, 64, dilation=1)
    x_tcn = tcn_block(x_tcn, 128, dilation=2)
    x_tcn = layers.GlobalAveragePooling1D()(x_tcn)
    z_res = layers.Dense(L // 2, name="z_res")(x_tcn)
    
    z_combined = layers.Concatenate(name="z_combined")([z_sys, z_res])
    return models.Model([inputs_sys, inputs_res], [z_sys, z_res, z_combined])


def build_dual_decoder(feat_sys, feat_res, output_steps, config):
    L = config.get("latent_dim", 512)
    drop_rate = config.get("dropout", 0.1)
    
    z_sys_in = layers.Input(shape=(L // 2,))
    z_res_in = layers.Input(shape=(L // 2,))
    
    # --- Decoder A: Physics Path (HNN) ---
    x_s = layers.Dense(output_steps * 64, activation='tanh', 
                       kernel_regularizer=regularizers.l2(REG))(z_sys_in)
    x_s = layers.Reshape((output_steps, 64))(x_s)
    if feat_sys > 0:
        x_s = layers.Conv1D(feat_sys, 1, padding='same', 
                        kernel_regularizer=regularizers.l2(REG))(x_s)
        out_sys = layers.Activation('linear', name='out_phy')(x_s)
    else:
        out_sys = layers.Lambda(lambda x: tf.zeros((tf.shape(x)[0], output_steps, 0)), name='out_phy')(x_s) 
    
    # --- Decoder B: TCN Path (Residual) ---
    x_r = layers.Dense(output_steps * 64, activation='relu', 
                       kernel_regularizer=regularizers.l2(REG))(z_res_in)
    x_r = layers.Reshape((output_steps, 64))(x_r)
    # Mirroring the TCN block structure in the decoder
    x_r = tcn_block(x_r, 128, dilation=2, dropout=drop_rate)
    x_r = layers.Conv1D(feat_res, 1, padding='same', 
                        kernel_regularizer=regularizers.l2(REG))(x_r)
    out_res = layers.Activation('linear', name='out_res')(x_r)
    
    return models.Model([z_sys_in, z_res_in], [out_sys, out_res])


def build_discriminator(input_dim):
    # ACAE Discriminator uses dropout of 0.5 
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dropout(0.5)(x) 
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation=None)(x) 
    return models.Model(inputs, outputs)