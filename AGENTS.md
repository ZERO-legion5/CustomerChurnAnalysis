# Agent Guidelines: Customer Churn Analysis

This document provides essential information for autonomous agents working on the Customer Churn Analysis repository. Adhere to these guidelines to maintain consistency, quality, and project integrity.

## 1. Project Overview
ChurnGuard is an AI-powered Customer Intelligence & Retention Platform. It predicts bank customer churn using an Artificial Neural Network (ANN), provides explainable AI (XAI) via SHAP, and generates personalized retention strategies using Google Gemini.

## 2. Setup & Environment

### Dependencies
- Install via pip: `pandas`, `numpy`, `matplotlib`, `seaborn`, `tensorflow`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `flask`, `joblib`, `imbalanced-learn`, `shap`, `reportlab`, `google-genai`, `python-dotenv`.
- **API Key**: Ensure `GEMINI_API_KEY` is available in your environment or a `.env` file.

### Execution
- **Run Web App**: `python app.py` (Flask starts on http://127.0.0.1:5000)
- **Train/Export Model**: `python export_model.py` (Generates `ann_model.h5`, `scaler.joblib`, `cluster_model.joblib`, etc.)
- **Execute Notebook**: `jupyter nbconvert --to notebook --execute main.ipynb`

## 3. Development Workflow & Quality

### Build & Linting
- **Linter**: Use `ruff check .` (preferred) or `flake8 .`.
- **Type Checking**: Use `mypy .` to verify type hints.
- **Formatting**: Adhere to PEP 8 standards.

### Testing
- **Framework**: Use `pytest`.
- **Run All Tests**: `pytest`
- **Run Single File**: `pytest tests/test_prediction.py`
- **Run Single Test**: `pytest tests/test_prediction.py::test_model_output`
- **Note**: If the `tests/` directory does not exist, create it before adding tests.

## 4. Code Style & Conventions

### Python Guidelines
- **Naming Conventions**:
  - Functions/Variables: `snake_case` (e.g., `get_prediction_score`)
  - Classes: `PascalCase` (e.g., `RetentionReporter`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `CHURN_THRESHOLD = 0.5`)
- **Imports Order**:
  1. Standard library (`os`, `io`, `sqlite3`)
  2. Third-party packages (`pandas`, `tensorflow`, `flask`)
  3. Local modules
- **Typing**: Mandatory for all new functions. Use `from typing import List, Dict, Any, Optional`.
  ```python
  def process_features(raw_data: Dict[str, Any]) -> np.ndarray:
      ...
  ```
- **Error Handling**: Always use `try...except` blocks in routes and data pipelines. Flask responses must return a `success` boolean.
  ```python
  try:
      return jsonify({'success': True, 'data': result})
  except Exception as e:
      app.logger.error(f"Prediction failed: {e}")
      return jsonify({'success': False, 'error': str(e)}), 500
  ```

### ML/Data Guidelines
- **Reproducibility**: Set `random_state=0` for all stochastic operations (splits, SMOTE, K-Means).
- **Feature Pipeline**:
  - `Geography`: France=0, Germany=1, Spain=2.
  - `Gender`: Female=0, Male=1.
  - **Scaling**: Use `MinMaxScaler` via `scaler.joblib`.
- **XAI**: Every prediction should include SHAP values to explain the "why" behind the result.

### Web & Frontend Guidelines
- **Styling**: Bootstrap 5 with a dark-themed custom CSS (`static/css/style.css`).
- **JS Flow**: Use the `fetch` API for all backend communication. Never use full page refreshes for analysis.
- **UI/UX**: Provide immediate visual feedback (spinners, disabled buttons) during asynchronous tasks.

## 5. Directory Structure
- `/`: Root for execution scripts (`app.py`, `export_model.py`) and data (`churn.csv`).
- `/templates`: Jinja2 templates (`base.html`, `index.html`, etc.).
- `/static`: Assets including `css/style.css`.
- `/tests`: (Planned) Pytest suites.
- `/catboost_info`: Metadata from training runs.

## 6. Implementation Specifications

### Database (SQLite)
- **File**: `retention.db`
- **Table**: `strategies`
- **Schema**: `id`, `customer_data` (JSON text), `prediction`, `strategy` (AI text), `timestamp`.

### Gemini AI Integration
- Use `google.genai` client.
- **Models**: `gemini-2.0-flash` for chat/general queries, `gemini-1.5-flash` or higher for complex strategy generation.
- **Prompts**: Maintain the professional, data-driven "ChurnGuard" persona.

### Model Architecture
- **ANN**: 10 inputs -> 32 (ReLU) -> 128 (ReLU, Drop 0.2) -> 64 (ReLU, Drop 0.2) -> 32 (Tanh) -> 1 (Sigmoid).
- **Clustering**: K-Means with 4 clusters for persona identification.

## 7. Operational Protocols
- **Sync Rule**: If you modify `export_model.py` or the preprocessing logic, you MUST run `python export_model.py` to update the joblib/h5 files used by `app.py`.
- **Safe Editing**: Read the entire file before applying `edit` or `write`. Preserve existing indentation and template tags.
- **Secret Safety**: Never hardcode API keys. Use `os.environ.get('GEMINI_API_KEY')`.
- **Dependency Management**: If you add a new library, update this document and the `Setup` section in `README.md`.
- **Communication**: Be concise in CLI; focus on task completion and verification status.

## 8. Roadmap & Agent Tasks
- **Persistence**: Migration from SQLite to a more robust storage if history grows.
- **XAI Expansion**: Add global SHAP plots to the Dashboard.
- **Persona Refining**: Improve the naming and logic of the 4 customer clusters.
- **Testing**: Implement a full test suite in `tests/` covering model inference and Flask routes.
- **CI/CD**: Add a GitHub Action to run `ruff` and `pytest` on every push.
- **Performance**: Optimize SHAP calculation speed (currently use `nsamples=100`).

## 9. Common Issues & Troubleshooting
- **Missing Models**: If `ann_model.h5` is missing, run `export_model.py`.
- **Gemini Errors**: Check API key quota and model availability (prefer `flash` models for speed).
- **SQLite Locks**: Avoid long-running transactions to prevent `database is locked` errors in Flask.
- **Scaling Mismatch**: Ensure `scaler.joblib` and `cluster_scaler.joblib` are used for their respective purposes.

---
*Last Updated: January 2026*
