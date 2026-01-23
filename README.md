# Customer Churn Analysis

This project aims to predict customer churn using various machine learning and deep learning models. The analysis is performed on a customer dataset, where features like credit score, geography, age, and balance are used to predict whether a customer will exit (churn).

## Models Included

- **Artificial Neural Network (ANN):** A multi-layer perceptron built with TensorFlow/Keras.
- **Random Forest:** An ensemble of decision trees.
- **XGBoost:** An optimized gradient boosting library.
- **LightGBM:** A fast, high-performance gradient boosting framework.
- **CatBoost:** A gradient boosting library that handles categorical features automatically.
- **Support Vector Machine (SVM):** A classic classifier with RBF kernel.
- **Logistic Regression:** A baseline linear model for classification.

## Flask Application

A professional Flask web application is included for real-time model deployment. It allows end users to input customer data and receive an instant prediction on churn probability.

### Running the App

1. Ensure all dependencies are installed:
   ```bash
   pip install pandas numpy tensorflow scikit-learn flask joblib
   ```
2. Export the latest model and scaler (if not already present):
   ```bash
   python export_model.py
   ```
3. Start the Flask server:
   ```bash
   python app.py
   ```
4. Open your browser and navigate to `http://127.0.0.1:5000`.

## Metrics and Evaluation

The models are evaluated based on several metrics to ensure a comprehensive assessment:
- **Accuracy:** Overall correctness of the model.
- **F1 Score (Macro):** Balanced measure for multi-class/imbalanced data.
- **AUC ROC:** Ability of the model to distinguish between classes.
- **Confusion Matrix:** Detailed breakdown of predictions.

## Leaderboard and Best Model

The project automatically compares all trained models and outputs a leaderboard. The best-performing model is identified based on the highest F1 Score, and its full set of metrics is displayed at the end of the analysis.

## Features & Intelligence Suite

- **Supervised Learning (ANN)**: High-accuracy churn prediction trained with SMOTE to handle class imbalance.
- **Explainable AI (SHAP)**: Individualized risk factor analysis for every prediction, showing exactly *why* a customer might leave.
- **Unsupervised Learning (K-Means)**: Automatic customer segmentation into behavioral personas, visualized in a live 3D cluster map.
- **Generative AI (Gemini)**: Integration with Google's `gemini-2.0-flash` to generate personalized retention strategies based on AI insights.
- **Business Dashboard**: Interactive macro-level analytics using Plotly.js.
- **Data Persistence**: Analysis history and retention plans saved in a local SQLite database.

## Getting Started

1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn tensorflow scikit-learn xgboost lightgbm catboost flask joblib imbalanced-learn shap reportlab google-genai
   ```
2. Set your **Gemini API Key** in your environment:
   ```bash
   set GEMINI_API_KEY=your_api_key_here  # Windows
   export GEMINI_API_KEY=your_api_key_here  # Linux/Mac
   ```
3. Export the models:
   ```bash
   python export_model.py
   ```
4. Start the platform:
   ```bash
   python app.py
   ```
