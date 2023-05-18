from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text


def load_data(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8')

    lines = text.splitlines()
    split_lines = [line.split('\t') for line in lines]

    context = []
    target = []

    for split_line in split_lines:
        target.append(split_line[0].strip())
        context.append(split_line[1].strip())

    context = np.array(context)
    target = np.array(target)

    return target, context

def tf_lower_and_split_punct(text):
    text = tf.strings.regex_replace(text, '[łŁ]', 'l')
    text = tf.strings.regex_replace(text, '[ąĄ]', 'a')
    text = tf.strings.regex_replace(text, '[ćĆ]', 'c')
    text = tf.strings.regex_replace(text, '[ęĘ]', 'e')
    text = tf.strings.regex_replace(text, '[ńŃ]', 'n')
    text = tf.strings.regex_replace(text, '[óÓ]', 'o')
    text = tf.strings.regex_replace(text, '[śŚ]', 's')
    text = tf.strings.regex_replace(text, '[źŹ]', 'z')
    text = tf.strings.regex_replace(text, '[żŻ]', 'z')

    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    
    text = tf.strings.regex_replace(text, '[^ a-z.?!,]', '')
    text = tf.strings.regex_replace(text, '[.?!,]', r' \0 ')
    
    text = tf.strings.strip(text)
    
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    
    return text

max_vocab_size = 5000

context_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size,
    ragged=True
)

target_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size,
    ragged=True
)

def process_text(context, target):
    context = context_text_processor(context).to_tensor()
    target = target_text_processor(target)
    targ_in = target[:,:-1].to_tensor()
    targ_out = target[:,1:].to_tensor()
    return (context, targ_in), targ_out