from tensorflow.keras.models import load_model
from model.preprocess import clean_text, prepare_input
from model.attention_layer import Attention

# Load model
model = load_model(
    "model/attention_bilstm.h5",
    custom_objects={"Attention": Attention}
)

def predict_sentiment(summary, review):
    combined = clean_text(summary + " " + review)
    padded = prepare_input(combined)

    prob = model.predict(padded)[0][0]
    sentiment = "positive" if prob > 0.5 else "negative"

    return sentiment, float(prob)