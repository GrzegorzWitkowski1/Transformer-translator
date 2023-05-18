import tensorflow as tf
from translator import Translator
from translator import print_translation
from model_training import context_text_processor
from model_training import target_text_processor
from sentence_parsing import split_sentences
from sentence_parsing import sentence_correction


def main():
    transformer = tf.saved_model.load('C:\\Users\\witko\\Documents\\Python\\Translator\\model_data\\transformer')
    
    translator = Translator(
        context_text_processor,
        target_text_processor,
        transformer
    )

    sentences = [
        "Cześć, jak się masz?",
        "Gdzie jest najbliższy sklep spożywczy?",
        "Jaki jest twój ulubiony kolor?",
        "Ile masz lat?",
        "Co chciałbyś zjeść na obiad?",
        "Gdzie mieszkasz?",
        "Czy możesz mi pomóc?",
        "Dziękuję Ci za pomoc!",
        "Jak się nazywasz?",
        "Czy lubisz sport?",
        "Czy możesz mi powiedzieć, która godzina?",
        "Którego języka obcego chciałbyś nauczyć się?",
        "Co robisz w wolnym czasie?",
        "Jakie są twoje plany na weekend?",
        "Gdzie można znaleźć dobre miejsce, aby wypić kawę?",
        "Jakie są twoje zainteresowania?",
        "Czy umiesz gotować?",
        "Jakie jest twoje ulubione danie?",
        "Co myślisz o polityce?",
        "Czy masz rodzeństwo?"
    ]

    for sentence in sentences:
        translated_text = translator(tf.constant(sentence))
        print_translation(sentence, translated_text)
    
    text = "" # jakiś tekst

    sentences = split_sentences(text)

    translated_string = ""

    for sentence in sentences:
        translated_string += sentence_correction(translator(tf.constant(sentence))) + " "

    print(text)
    print(translated_string)


if __name__ == '__main__':
    main()