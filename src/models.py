import tensorflow as tf
from tensorflow.keras import layers, models


def residual_block(filters, downsample=False):
    def block(x_input):
        stride = 2 if downsample else 1

        x = layers.Conv1D(filters, 3, padding='same', strides=stride, activation='relu')(x_input)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters, 3, padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)

        # Downsample shortcut if needed
        if downsample or x_input.shape[-1] != filters:
            x_input = layers.Conv1D(filters, 1, strides=stride, padding='same')(x_input)

        out = layers.Add()([x, x_input])
        return layers.Activation('relu')(out)
    return block


def build_encoder(input_shape=(64, 38), latent_dim=256):
    inputs = layers.Input(shape=input_shape)

    x = residual_block(64, downsample=True)(inputs)   # (32, 64)
    x = residual_block(64)(x)                          # (32, 64)
    x = residual_block(128, downsample=True)(x)        # (16, 128)
    x = residual_block(128)(x)                         # (16, 128)

    x = layers.GlobalAveragePooling1D()(x)             # (128,)
    z = layers.Dense(latent_dim)(x)                    # (256,)

    return models.Model(inputs, z, name="Encoder")


def build_decoder(latent_dim=256, output_shape=(64, 38)):
    time_steps, features = output_shape
    inputs = layers.Input(shape=(latent_dim,))

    x = layers.Dense(16 * 128, activation='relu')(inputs)
    x = layers.Reshape((16, 128))(x)

    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)

    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)

    outputs = layers.Conv1D(features, 3, padding='same')(x)
    return models.Model(inputs, outputs, name="Decoder")


def build_discriminator(latent_dim=256):
    inputs = layers.Input(shape=(latent_dim,))

    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(2, activation='sigmoid')(x)  # [label, proportion]
    return models.Model(inputs, outputs, name="Discriminator")
