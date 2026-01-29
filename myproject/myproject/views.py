# views.py
import os
import joblib
import numpy as np

from django.shortcuts import render
from django.conf import settings

# ======================
# Classical ML Model Loading
# ======================

MODELS_DIR = os.path.join(settings.BASE_DIR, "model")   # classical models (.pkl)
available_models = {}

for model_file in os.listdir(MODELS_DIR):
    if model_file.endswith(".pkl"):
        model_name = model_file.replace(".pkl", "")
        model_path = os.path.join(MODELS_DIR, model_file)

        try:
            model_data = joblib.load(model_path)
            available_models[model_name] = model_data
            print(f"✅ Loaded ML model: {model_name}")

        except Exception as e:
            print(f"❌ Failed to load {model_file}: {e}")

# ======================
# Deep Learning Model Loading
# ======================

from tensorflow.keras.models import load_model
from dl_model.attention_layer import Attention
from dl_model.preprocess import clean_text, prepare_input

DL_MODEL_PATH = os.path.join(settings.BASE_DIR, "dl_model", "attention_bilstm.h5")
DL_TOKENIZER_PATH = os.path.join(settings.BASE_DIR, "dl_model", "tokenizer.pkl")

dl_model = None
dl_model_loaded = False

try:
    dl_model = load_model(
        DL_MODEL_PATH,
        custom_objects={"Attention": Attention}
    )
    dl_model_loaded = True
    print("🚀 Loaded Deep Learning Model: Attention BiLSTM")

except Exception as e:
    print(f"❌ Failed to load DL model: {e}")


# Manually register DL model into model list
available_models["deep_learning"] = "Attention BiLSTM"


# ======================
# Views
# ======================

def home(request):
    return render(request, "home.html")


def predict(request):
    result = None
    text = ""
    selected_model = "logistic"  # default model

    if request.method == "POST":
        text = request.POST.get("text", "").strip()
        selected_model = request.POST.get("model", "logistic")

        if text and selected_model in available_models:

            try:
                # ============================
                # Case 1: Deep Learning Model
                # ============================
                if selected_model == "deep_learning":

                    if not dl_model_loaded:
                        raise Exception("Deep Learning model not loaded.")

                    cleaned = clean_text(text)
                    padded = prepare_input(cleaned)
                    prob = dl_model.predict(padded)[0][0]

                    pred = 1 if prob > 0.5 else 0
                    sentiment_map = {
                        0: "Negative",
                        1: "Positive"
                    }
                    sentiment = sentiment_map[pred]

                # ============================
                # Case 2: Classical ML Models
                # ============================
                else:
                    model_data = available_models[selected_model]
                    model = model_data["model"]
                    vectorizer = model_data["vectorizer"]

                    vec = vectorizer.transform([text])
                    pred = model.predict(vec)[0]

                    sentiment_map = {
                        0: "Negative",
                        1: "Positive",
                        2: "Neutral"
                    }

                    sentiment = sentiment_map.get(pred, "Unknown")

                # ============================
                # Build Result Dictionary
                # ============================
                result = {
                    "sentiment": sentiment,
                    "model_name": selected_model.replace("_", " ").title(),
                    "text_preview": text[:100] + "..." if len(text) > 100 else text
                }

                print(f"✅ Prediction successful: {sentiment}")

            except Exception as e:
                print(f"❌ Prediction error: {e}")

                result = {
                    "sentiment": "Error",
                    "model_name": selected_model.replace("_", " ").title(),
                    "text_preview": text[:100] + "..." if len(text) > 100 else text,
                    "error": str(e),
                }

        elif not text:
            print("❌ No text entered")

        elif selected_model not in available_models:
            print(f"❌ Model '{selected_model}' not found")

    # Template context
    context = {
        "result": result,
        "text": text,
        "selected_model": selected_model,
        "available_models": list(available_models.keys()),
    }

    return render(request, "predict.html", context)


def about(request):
    return render(request, "about.html")