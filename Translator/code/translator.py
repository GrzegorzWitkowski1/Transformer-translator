import tensorflow as tf


MAX_TOKENS = 20

class Translator(tf.Module):
    def __init__(self, context_text_processor, target_text_processor, transformer):
        self.context_tokenizer = context_text_processor
        self.target_tokenizer = target_text_processor
        self.transformer = transformer

    def __call__(self, sentence, max_length=MAX_TOKENS):
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.context_tokenizer(sentence).to_tensor()

        encoder_input = sentence

        start_end = self.target_tokenizer([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)

            # Select the last token from the `seq_len` dimension.
            predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

            predicted_id = tf.argmax(predictions, axis=-1)

            # Terminate if the predicted ID is the end token.
            if tf.reduce_all(tf.equal(predicted_id, end)):
                break

            # Write the predicted ID to the output array.
            output_array = output_array.write(i + 1, predicted_id[0])

        # Convert the output array to a tensor and remove the batch dimension.
        output = output_array.stack()
        output = tf.squeeze(output, axis=1)

        # Convert token IDs to text.
        predicted_sentence = [self.target_tokenizer.get_vocabulary()[token_id] for token_id in output.numpy()]

        # Remove the start and end tokens from the predicted sentence.
        predicted_sentence = predicted_sentence[1:-1]

        return predicted_sentence
    
def print_translation(sentence, tokens):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {" ".join(tokens)}')
