import tensorflow as tf

class ACEncoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(ACEncoder, self).__init__()
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(latent_dim)

    def call(self, x):
        x = self.lstm(x)
        x = self.flatten(x)
        return self.dense(x)
