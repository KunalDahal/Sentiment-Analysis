# Sentiment Analysis Web Application

A Django-based sentiment analysis application with beautiful UI and machine learning integration.

## Project Structure

```
myproject/
├── models/
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   └── linear_svm.pkl
├── templates/
│   ├── home.html
│   ├── predict.html
│   └── about.html
├── views.py
├── urls.py
└── settings.py
```

## Setup Instructions

### 1. Install Required Packages

```bash
pip install django scikit-learn numpy
```

### 2. Place Model Files

Ensure your trained pickle model files are in the `myproject/models/` directory:
- `logistic_regression.pkl`
- `naive_bayes.pkl`
- `linear_svm.pkl`

### 3. Configure Django URLs

In your main project `urls.py`, include the app URLs:

```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('your_app_name.urls')),
]
```

### 4. Update Settings

In `settings.py`, add your app to `INSTALLED_APPS`:

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'your_app_name',  # Add your app name here
]
```

Configure templates directory:

```python
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],  # Add this
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
```

### 5. Place HTML Templates

Place the provided HTML files in your `templates/` directory:
- `home.html` - Landing page
- `predict.html` - Prediction interface
- `about.html` - About page (you'll need to create this)

### 6. Copy View Logic

Copy the `views.py` code to your app's views file.

### 7. Copy URL Configuration

Copy the `urls.py` code to your app's urls file.

## How It Works

### Model Loading

The application uses lazy loading for ML models:

```python
MODEL_PATHS = {
    'logistic': os.path.join(BASE_DIR, 'models', 'logistic_regression.pkl'),
    'naive_bayes': os.path.join(BASE_DIR, 'models', 'naive_bayes.pkl'),
    'svm': os.path.join(BASE_DIR, 'models', 'linear_svm.pkl'),
}
```

Models are loaded only when first requested and cached for subsequent use.

### Prediction Flow

1. User enters text and selects a model (Logistic Regression, Naive Bayes, or Linear SVM)
2. Form is submitted via POST to `/predict/`
3. Django view:
   - Validates input
   - Loads the selected model from pickle file
   - Makes prediction using `model.predict([text])`
   - Calculates confidence score using `predict_proba()` or `decision_function()`
   - Determines sentiment (Positive/Negative)
   - Returns results to template
4. Template displays:
   - Sentiment result (Positive/Negative)
   - Confidence score with animated progress bar
   - Model name used
   - Original text

### Model Requirements

Your pickle files should contain sklearn models with these methods:
- `predict(X)` - Required: Returns predictions
- `predict_proba(X)` - Optional: Returns probability estimates
- `decision_function(X)` - Optional: Returns decision scores (for SVM)

The models should accept a list of text strings as input.

### Expected Model Format

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Example: Your models should be pipelines or have fitted vectorizers
model = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# After training
import pickle
with open('logistic_regression.pkl', 'wb') as f:
    pickle.dump(model, f)
```

## Running the Application

```bash
# Run migrations (first time only)
python manage.py migrate

# Start development server
python manage.py runserver

# Access the application
# Open browser: http://127.0.0.1:8000/
```

## Features

### Home Page (`/`)
- Modern animated background with orange/yellow gradient theme
- Feature highlights
- Statistics display
- Navigation to Predict and About pages

### Predict Page (`/predict/`)
- Three model options:
  - Logistic Regression (94.2% accuracy)
  - Naive Bayes (92.8% accuracy)
  - Linear SVM (95.6% accuracy)
- Text input with character counter (5000 max)
- Quick example buttons for testing
- Real-time sentiment prediction
- Animated confidence score display
- Keyboard shortcut (Ctrl+Enter to submit)

### UI Features
- Responsive design (mobile-friendly)
- Smooth animations and transitions
- Glassmorphism effects
- Particle background animations
- Hover effects on all interactive elements
- Form validation
- Error handling with user-friendly messages

## Customization

### Changing Colors

Edit CSS variables in the `<style>` section:

```css
:root {
    --primary-orange: #ff9f43;
    --accent-yellow: #ffd32a;
    --deep-orange: #ff6b35;
    /* ... modify as needed */
}
```

### Adding More Models

1. Save your trained model pickle file in `models/` directory
2. Add model path in `views.py`:

```python
MODEL_PATHS = {
    'logistic': os.path.join(BASE_DIR, 'models', 'logistic_regression.pkl'),
    'naive_bayes': os.path.join(BASE_DIR, 'models', 'naive_bayes.pkl'),
    'svm': os.path.join(BASE_DIR, 'models', 'linear_svm.pkl'),
    'new_model': os.path.join(BASE_DIR, 'models', 'your_new_model.pkl'),  # Add here
}
```

3. Add model card in `predict.html`:

```html
<label class="model-option">
    <input type="radio" name="model" value="new_model">
    <div class="model-card">
        <div class="model-name">Your Model Name</div>
        <div class="model-desc">Description of your model</div>
        <div class="model-stats">
            <div class="stat">
                <span class="stat-label">Accuracy</span>
                <span class="stat-value">XX.X%</span>
            </div>
        </div>
    </div>
</label>
```

4. Add to model names dictionary in `views.py`:

```python
model_names = {
    'logistic': 'Logistic Regression',
    'naive_bayes': 'Naive Bayes',
    'svm': 'Linear SVM',
    'new_model': 'Your Model Name',
}
```

## Troubleshooting

### Models Not Loading

**Error**: `Model file not found`

**Solution**: Check that:
- Model files are in `myproject/models/` directory
- File names match exactly (case-sensitive)
- Files have `.pkl` extension
- BASE_DIR path is correct in views.py

### Prediction Errors

**Error**: `predict() takes X positional arguments but Y were given`

**Solution**: Your model expects different input format. Ensure models accept list of strings:
```python
model.predict([text])  # Not model.predict(text)
```

### Import Errors

**Error**: `No module named 'sklearn'`

**Solution**: Install scikit-learn:
```bash
pip install scikit-learn
```

### Template Not Found

**Error**: `TemplateDoesNotExist`

**Solution**: 
- Check `TEMPLATES` configuration in settings.py
- Ensure templates are in correct directory
- Verify app is in `INSTALLED_APPS`

## Internship Project Information

**For**: Euphoria GENX  
**Type**: Machine Learning Web Application  
**Technologies**: Django, Python, HTML/CSS, JavaScript, Scikit-learn  

## License

This project is created for educational/internship purposes.

## Support

For questions or issues, please check:
1. Django documentation: https://docs.djangoproject.com/
2. Scikit-learn documentation: https://scikit-learn.org/
3. Python pickle documentation: https://docs.python.org/3/library/pickle.html