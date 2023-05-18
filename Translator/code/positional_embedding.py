import tensorflow as tf
import numpy as np


def positional_encoding(length: int, depth: int) -> tf.Tensor:
    '''
    Generates positional encodings for a given length and depth.

    Positional encodings are used in transformer-based models to incorporate
    positional information into the input data.

    Args:
        length (int): The number of positions or elements in the positional encoding.
        depth (int): The depth or dimensionality of the positional encoding.

    Returns:
        tf.Tensor: A tensor containing the positional encodings.
                   Shape: (length, depth)
                   Data type: tf.float32
    '''
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth

    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis = -1
    )

    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            d_model,
            mask_zero=True
        )

        self.pos_encoding = positional_encoding(
            length=2048,
            depth=d_model
        )

    def call(self, x):
        length = tf.shape(x)[1]
        
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        
        return x

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)