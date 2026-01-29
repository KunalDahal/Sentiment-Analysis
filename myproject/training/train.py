import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM,
    Dropout, Dense
)

from model.attention_layer import Attention
from training.dataset import load_dataset

VOCAB_SIZE = 20000
MAX_LEN = 150
EMB_DIM = 128

def build_attention_bilstm():
    inputs = Input(shape=(MAX_LEN,))
    x = Embedding(VOCAB_SIZE, EMB_DIM)(inputs)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Attention()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


df = load_dataset("flipkart_reviews.csv")
texts = df["final_text"].values
labels = df["label"].values

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# Save tokenizer
with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

seq = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

X_train, X_val, y_train, y_val = train_test_split(
    padded, labels, test_size=0.2, stratify=labels, random_state=42
)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: w for i, w in enumerate(class_weights)}

model = build_attention_bilstm()
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=64,
    class_weight=class_weights,
    verbose=1
)

model.save("model/attention_bilstm.h5")
print("Model saved successfully!")