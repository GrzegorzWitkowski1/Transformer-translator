import re


def split_sentences(text: str) -> list:
    # Define the regex pattern to match sentence boundaries
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'

    # Split the text into sentences using the regex pattern
    sentences = re.split(pattern, text)

    return sentences

def sentence_correction(text: str) -> str:
    text = text.capitalize()
    text = re.sub(r'\s+([.!?])', r'\1', text)
    
    return text