import tensorflow as tf
import numpy as np
from data_processing import load_data
from data_processing import context_text_processor
from data_processing import target_text_processor
from data_processing import process_text
from transformer import Transformer
from custom_schedule import CustomSchedule
from training_metrics import masked_accuracy
from training_metrics import masked_loss


def print_translation(sentence, tokens):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {" ".join(tokens)}')


def train_model():
    target_raw, context_raw = load_data("C:\\Users\\witko\\Documents\\Python\\Translator\\model_data\\pol.txt")

    BUFFER_SIZE = len(context_raw)
    BATCH_SIZE = 64

    is_train = np.random.uniform(size=(len(target_raw),)) < 0.8

    train_raw = (
        tf.data.Dataset
            .from_tensor_slices((context_raw[is_train], target_raw[is_train]))
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
    )

    val_raw = (
        tf.data.Dataset
            .from_tensor_slices((context_raw[~is_train], target_raw[~is_train]))
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
    )

    context_text_processor.adapt(train_raw.map(lambda context, target: context))
    target_text_processor.adapt(train_raw.map(lambda context, target: target))

    train_ds = train_raw.map(process_text, tf.data.AUTOTUNE)
    val_ds = val_raw.map(process_text, tf.data.AUTOTUNE)

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=len(context_text_processor.get_vocabulary()),
        target_vocab_size=len(target_text_processor.get_vocabulary()),
        dropout_rate=dropout_rate
    )

    learning_rate = CustomSchedule(d_model)
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9, 
        beta_2=0.98, 
        epsilon=1e-9
    )

    transformer.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy]
    )

    transformer.fit(
        train_ds,
        epochs=20,
        validation_data=val_ds
    )