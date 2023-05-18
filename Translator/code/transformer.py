import tensorflow as tf
from encoder import Encoder
from decoder import Decoder


class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()

        self.encoder = Encoder(
            num_layers=num_layers, d_model=d_model,
            num_heads=num_heads, dff=dff,
            vocab_size=input_vocab_size,
            dropout_rate=dropout_rate
        )

        self.decoder = Decoder(
            num_layers=num_layers, d_model=d_model,
            num_heads=num_heads, dff=dff,
            vocab_size=target_vocab_size,
            dropout_rate=dropout_rate
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        context, x = inputs

        context = self.encoder(context)

        x = self.decoder(x, context)

        logits = self.final_layer(x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits