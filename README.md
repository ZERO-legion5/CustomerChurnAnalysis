# ChurnGuard: Customer Intelligence & Retention Platform

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Flask-red.svg)
![AI](https://img.shields.io/badge/AI-Gemini%202.0-green.svg)

**ChurnGuard** is an AI-powered end-to-end platform designed to help banks identify at-risk customers and implement data-driven retention strategies. By combining Deep Learning, Explainable AI (XAI), and Generative AI, ChurnGuard transforms raw data into actionable business intelligence.

---

## 🚀 Key Features

- **Predictive Analytics**: High-precision churn prediction using an Artificial Neural Network (ANN) optimized with SMOTE.
- **Explainable AI (SHAP)**: Individualized risk factor analysis. Understand exactly *why* the model predicts a specific outcome for every customer.
- **Customer Personas**: Automatic segmentation using K-Means clustering, visualized through an interactive 3D behavioral map.
- **AI Retention Strategies**: Personalized email and retention plan generation powered by **Google Gemini 2.0**, tailored to specific risk factors.
- **Intelligence Dashboard**: Macro-level insights into churn trends across geography, age, and account balance.
- **Retention Lab**: A sandbox to test retention offers and save successful strategies to the local intelligence database.

---

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow/Keras, Scikit-Learn, SHAP, Imbalanced-Learn
- **Generative AI**: Google GenAI (Gemini 2.0 Flash)
- **Database**: SQLite (SQLAlchemy)
- **Frontend**: Bootstrap 5, FontAwesome, Plotly.js, Three.js (for 3D viz)
- **Reporting**: ReportLab (PDF Generation)

---

## 📥 Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/CustomerChurnAnalysis.git
   cd CustomerChurnAnalysis
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If requirements.txt is missing, install manually:*
   `pip install pandas numpy matplotlib seaborn tensorflow scikit-learn xgboost lightgbm catboost flask joblib imbalanced-learn shap reportlab google-genai python-dotenv`

3. **Configure Environment**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

---

## ⚙️ Usage

### 1. Model Preparation
If the model files (`ann_model.h5`, `scaler.joblib`, etc.) are not present, run the export script:
```bash
python export_model.py
```

### 2. Launch the Platform
Start the Flask application:
```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your browser.

### 3. Exploratory Analysis
The original data science workflow and model comparison leaderboard can be found in `main.ipynb`.

---

## 📊 Model Architecture

The core of ChurnGuard is a multi-layer Artificial Neural Network:
- **Input**: 10 engineered features
- **Processing**: 5 layers (ReLU & Tanh activations) with Dropout regularization
- **Output**: Sigmoid activation for binary churn probability

---

## 📁 Project Structure

- `app.py`: Main Flask application and API routes.
- `export_model.py`: Training and artifact serialization pipeline.
- `main.ipynb`: Research, EDA, and model benchmarking.
- `templates/`: Jinja2 templates for the responsive UI.
- `retention.db`: Persistent storage for AI-generated strategies.
