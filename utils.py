import re

def preprocess_text(text):
    """
    Preprocess the input text by converting to lowercase and removing punctuation.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text
