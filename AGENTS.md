# Agent Guidelines: Customer Churn Analysis

This document provides essential information for autonomous agents working on this repository. Adhere to these guidelines to maintain consistency and quality.

## 1. Project Overview
This is a Customer Intelligence & Retention Platform that uses machine learning to predict bank customer churn. It includes a data analysis notebook, a model training pipeline, and a Flask-based web application.

## 2. Environment & Commands

### Setup
- Install dependencies: `pip install pandas numpy matplotlib seaborn tensorflow scikit-learn xgboost lightgbm catboost flask joblib imbalanced-learn shap reportlab ollama`
- Ensure Ollama is running with `llama3.2:3b`.

### Execution
- **Run Web App**: `python app.py` (Starts Flask on http://127.0.0.1:5000)
- **Train/Export Model**: `python export_model.py` (Generates `ann_model.h5` and `scaler.joblib`)
- **Run Notebook**: Use `jupyter nbconvert --to notebook --execute main.ipynb` or open in a Jupyter environment.

### Testing & Quality
- **Linter**: `flake8 .` or `ruff check .`
- **Tests**: Currently no tests are implemented. When adding them:
  - Use **pytest**: `pytest`
  - Single test: `pytest tests/test_file.py::test_function_name`
- **Type Checking**: `mypy .`

## 3. Code Style & Conventions

### Python Guidelines
- **Standard**: Follow PEP 8.
- **Naming**: 
  - Variables/Functions: `snake_case` (e.g., `calculate_risk`)
  - Classes: `PascalCase` (e.g., `ChurnPredictor`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_THRESHOLD = 0.5`)
- **Imports**:
  1. Standard library (e.g., `import os`)
  2. Third-party libraries (e.g., `import pandas as pd`)
  3. Local modules (e.g., `from app import app`)
- **Typing**: Use type hints for all function signatures.
  ```python
  def predict_churn(data: list[float]) -> float:
      ...
  ```
- **Error Handling**: Use explicit `try...except` blocks. In Flask routes, always return JSON with a `success` flag.
  ```python
  try:
      # logic
      return jsonify({'success': True, 'data': result})
  except Exception as e:
      return jsonify({'success': False, 'error': str(e)}), 500
  ```

### ML/Data Guidelines
- **Reproducibility**: Always set a `random_state` (default: 0) when splitting data or initializing models.
- **Preprocessing**: Ensure features are scaled using the same `scaler.joblib` used during training.
- **Explainability**: Prioritize SHAP or similar libraries for model interpretation in future features.

### Web/Frontend Guidelines
- **Frontend**: Use Bootstrap 5 for styling. Ensure responsive design.
- **Interactions**: Prefer Asynchronous requests (Fetch API) over full page refreshes.
- **Feedback**: Provide loading states (spinners) and clear success/error messages.

## 4. Directory Structure
- `/`: Root directory containing main scripts and notebooks.
- `/templates`: HTML templates for the Flask application.
- `/static`: (Planned) For CSS, JavaScript, and Image assets.
- `/models`: (Planned) To store versioned model files.

## 5. Model Architecture Details
The primary model is an Artificial Neural Network (ANN) with the following structure:
- **Input Layer**: 10 features (after dropping RowNumber, CustomerId, Surname).
- **Hidden Layer 1**: 32 units, ReLU activation.
- **Hidden Layer 2**: 128 units, ReLU activation.
- **Hidden Layer 3**: 64 units, ReLU activation.
- **Hidden Layer 4**: 32 units, Tanh activation.
- **Output Layer**: 1 unit, Sigmoid activation.
- **Regularization**: Dropout (0.2) applied after layers 2 and 3.

## 6. Implementation Specifications

### Handling Categorical Data
- **Geography**: Encoded as France=0, Germany=1, Spain=2.
- **Gender**: Encoded as Female=0, Male=1.
- Agents must ensure manual mapping in `app.py` matches the `LabelEncoder` used in `export_model.py`.

### Feature Scaling
- All numerical inputs must be scaled using `MinMaxScaler`.
- The `scaler.joblib` file is the source of truth for scaling parameters.

### Asynchronous Flow
The frontend uses the `fetch` API to send JSON to `/predict`. Agents should:
1. Prevent default form submission.
2. Show a loading spinner on the button.
3. Update the `result-area` dynamically without page refresh.

## 7. Future Roadmap & Agent Tasks
When assigned tasks, agents should look to:
- **XAI**: Integrate SHAP to explain individual predictions.
- **Analytics**: Create a dashboard with charts for churn distribution.
- **Imbalance**: Implement SMOTE in the training pipeline to handle class imbalance.
- **Persistence**: Add a database (SQLite/PostgreSQL) to store prediction history.

## 8. Operational Protocols
- **File Access**: Always read a file before editing it to ensure context.
- **Changes**: Do not revert user changes or project conventions without explicit instruction.
- **Communication**: Be concise in CLI interactions; focus on results.
- **Verification**: After making changes to the model or preprocessing, always run `export_model.py` to ensure the app stays in sync with the training logic.
