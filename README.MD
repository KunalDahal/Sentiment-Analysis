# Sentiment Analysis Web Application

A machine learning and deep learning-powered web application for analyzing customer sentiment from Flipkart product reviews.

## Overview

This project classifies customer reviews into positive, negative, or neutral sentiments using advanced machine learning and deep learning algorithms. The system provides real-time sentiment predictions through an intuitive web interface built with Django.

## Dataset

The project uses the **Flipkart Product Customer Reviews Dataset** from Kaggle, which contains real customer product reviews with sentiment labels from a major e-commerce platform.

**Dataset Link:** [Flipkart Product Customer Reviews on Kaggle](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset)

## Project Workflow

1. **Data Collection** - Imported Flipkart customer reviews dataset from Kaggle
2. **Data Cleaning** - Handled missing values, removed duplicates, and filtered invalid entries
3. **Text Preprocessing** - Applied lowercasing, removed special characters, performed tokenization and vectorization
4. **Exploratory Data Analysis** - Analyzed sentiment distribution, review patterns, and feature correlations
5. **Feature Engineering** - Extracted meaningful features using TF-IDF vectorization
6. **Model Development** - Trained both deep learning and machine learning models
7. **Web Application** - Deployed models through a Django web interface

## Models Implemented

### Deep Learning Model
- **BiLSTM (Bidirectional Long Short-Term Memory)** - Analyzes text in both forward and backward directions for comprehensive context understanding

### Machine Learning Models
- **Linear SVM (Support Vector Machine)** - Finds optimal hyperplane for sentiment classification
- **Logistic Regression** - Probabilistic approach for multi-class sentiment prediction
- **Naive Bayes** - Fast probabilistic classifier based on Bayes' theorem

## Technology Stack

### Frontend
- HTML5
- CSS3
- JavaScript
- Responsive Design

### Backend
- Python 3.x
- Django Framework
- Django Templates

### Machine Learning & Deep Learning
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy

### Data Visualization
- Matplotlib
- Seaborn
- Jupyter Notebooks

### Version Control
- Git
- GitHub

## Features

- Simple and intuitive user interface
- Real-time sentiment prediction on user input
- Multiple model selection for comparison
- Instant results with clear sentiment classification
- Responsive design for desktop and mobile devices
- High accuracy (>90%) on validation data

## Team

- **Abishek Prasad** - Deep Learning Model Developer (BiLSTM Implementation)
- **Kunal Dahal** - Machine Learning Model Developer (Linear SVM, Logistic Regression, Naive Bayes)
- **Natasha Rai** - Web Developer & Documentation (Backend & Frontend Development)
- **Riddhi Shree** - Web Developer & Documentation (Backend & Frontend Development)

## Project Purpose

This project demonstrates the practical application of AI in understanding customer feedback and extracting actionable insights from unstructured text data. It helps businesses to:

- Monitor customer satisfaction in real-time
- Identify product issues quickly
- Understand buying behavior and preferences
- Make informed business decisions based on customer feedback

## License

This project is open source and available for educational purposes.