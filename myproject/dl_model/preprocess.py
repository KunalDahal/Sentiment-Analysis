import re
import emoji
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 150

# Load tokenizer
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


def clean_text(text):
    if text is None:
        return ""
    text = str(text)

    text = re.sub(r"<.*?>", "", text)
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def combine_text(summary, review):
    summary = clean_text(summary)
    review = clean_text(review)
    return (summary + " " + review).strip()


def prepare_input(text):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=MAX_LEN, padding="post")