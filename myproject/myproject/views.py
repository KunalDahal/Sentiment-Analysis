# views.py - Simplified version
import joblib
import os
import numpy as np
from django.shortcuts import render
from django.conf import settings

# Load all available models
MODELS_DIR = os.path.join(settings.BASE_DIR, 'models')
available_models = {}

# Load all model files
for model_file in os.listdir(MODELS_DIR):
    if model_file.endswith('.pkl'):
        model_name = model_file.replace('.pkl', '')
        model_path = os.path.join(MODELS_DIR, model_file)
        try:
            model_data = joblib.load(model_path)
            available_models[model_name] = model_data
            print(f"✅ Loaded model: {model_name}")
        except Exception as e:
            print(f"❌ Failed to load {model_file}: {e}")

def home(request):
    return render(request, "home.html")

def predict(request):
    result = None
    text = ""
    selected_model = "logistic"  # Default model
    
    if request.method == "POST":
        text = request.POST.get("text", "").strip()
        selected_model = request.POST.get("model", "logistic")
        
        if text and selected_model in available_models:
            try:
                # Get the selected model and vectorizer
                model_data = available_models[selected_model]
                model = model_data['model']
                vectorizer = model_data['vectorizer']
                
                # Transform and predict
                vec = vectorizer.transform([text])
                pred = model.predict(vec)[0]
                
                # Map prediction to sentiment
                # 0 = Negative, 1 = Positive, 2 = Neutral
                sentiment_map = {
                    0: "Negative",
                    1: "Positive",
                    2: "Neutral"
                }
                
                sentiment = sentiment_map.get(pred, "Unknown")
                
                result = {
                    'sentiment': sentiment,
                    'model_name': selected_model.replace('_', ' ').title(),
                    'text_preview': text[:100] + "..." if len(text) > 100 else text
                }
                
                print(f"✅ Prediction successful: {sentiment}")
                
            except Exception as e:
                print(f"❌ Prediction error: {e}")
                result = {
                    'sentiment': 'Error',
                    'model_name': selected_model.replace('_', ' ').title(),
                    'text_preview': text[:100] + "..." if len(text) > 100 else text,
                    'error': str(e)
                }
        elif not text:
            print("❌ No text provided")
        elif selected_model not in available_models:
            print(f"❌ Model '{selected_model}' not found")
    
    # Prepare context for template
    context = {
        "result": result,
        "text": text,  # This preserves the input text after submission
        "selected_model": selected_model,
        "available_models": list(available_models.keys())
    }
    
    return render(request, "predict.html", context)

def about(request):
    return render(request, 'about.html')