from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import shap
import io
import sqlite3
import os
import json
import csv
import threading
from typing import Any, Dict, List, Optional
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve, f1_score, accuracy_score, recall_score, precision_score
from google import genai
from google.genai import types
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import inch
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# --- Load Models & Utilities ---
client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))
model = tf.keras.models.load_model('ann_model.h5')
scaler = joblib.load('scaler.joblib')
background_data = joblib.load('background_data.joblib')
explainer = shap.KernelExplainer(model.predict, background_data)

# Persona Models
kmeans = joblib.load('cluster_model.joblib')
cluster_scaler = joblib.load('cluster_scaler.joblib')

# Mappings
geo_map = {'France': 0, 'Germany': 1, 'Spain': 2}
gender_map = {'Female': 0, 'Male': 1}
feature_names = ['Credit Score', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'Number of Products', 'Has Credit Card', 'Is Active Member', 'Estimated Salary']

PERSONA_NAMES = {
    0: 'Dormant Affluents',
    1: 'Young Transients',
    2: 'Loyal Core',
    3: 'High-Value Actives'
}
PERSONA_ICONS = {
    0: 'fa-bed',
    1: 'fa-running',
    2: 'fa-shield-alt',
    3: 'fa-crown'
}
PERSONA_DESC = {
    0: 'Older customers with high balances but low engagement. High churn risk due to inactivity.',
    1: 'Younger customers with short tenures and moderate balances. Sensitive to product value.',
    2: 'Mid-age customers with long tenure and strong credit. Most loyal segment.',
    3: 'High-balance, active members with premium credit scores. Highest lifetime value.'
}

# --- Database Initialization ---
def init_db():
    conn = sqlite3.connect('retention.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS strategies 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  customer_data TEXT, 
                  prediction TEXT, 
                  strategy TEXT, 
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# --- Pre-compute expensive model stats once at startup ---
MODEL_STATS_CACHE_FILE = 'model_stats_cache.json'
_model_stats_cache: Dict[str, Any] = {}
_model_stats_lock = threading.Lock()

def _build_model_stats():
    """Compute confusion matrix, ROC, PR, and global SHAP once and cache."""
    global _model_stats_cache
    df = pd.read_csv("churn.csv")
    feature_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                    'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    df_enc = df.copy()
    df_enc['Geography'] = df_enc['Geography'].map(geo_map)
    df_enc['Gender'] = df_enc['Gender'].map(gender_map)
    X = df_enc[feature_cols].values
    y = df_enc['Exited'].values
    X_scaled = scaler.transform(X)

    y_prob = model.predict(X_scaled, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    cm = confusion_matrix(y, y_pred).tolist()
    acc  = round(accuracy_score(y, y_pred) * 100, 2)
    f1   = round(f1_score(y, y_pred) * 100, 2)
    rec  = round(recall_score(y, y_pred) * 100, 2)
    prec = round(precision_score(y, y_pred) * 100, 2)
    auc  = round(roc_auc_score(y, y_prob) * 100, 2)

    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_data = {'fpr': fpr.tolist()[::10], 'tpr': tpr.tolist()[::10]}

    prec_vals, rec_vals, _ = precision_recall_curve(y, y_prob)
    pr_data = {'precision': prec_vals.tolist()[::10], 'recall': rec_vals.tolist()[::10]}

    # Global SHAP on 150 samples — use small background to avoid broadcast mismatch
    rng = np.random.RandomState(0)
    sample_idx = rng.choice(len(X_scaled), 150, replace=False)
    X_sample = X_scaled[sample_idx]
    # Subsample background to 50 rows to prevent (N,1) broadcast shape errors
    bg_idx = rng.choice(len(background_data), min(50, len(background_data)), replace=False)
    bg_sample = background_data[bg_idx] if hasattr(background_data, '__len__') else background_data
    local_explainer = shap.KernelExplainer(model.predict, bg_sample)
    shap_vals = local_explainer.shap_values(X_sample, nsamples=100)
    # shap_vals can be: list of arrays per class, single 2D array, or 3D array
    if isinstance(shap_vals, list):
        # list[0]=class0(Stay), list[1]=class1(Churn) — use churn class
        shap_arr = np.abs(np.array(shap_vals[-1]))  # last = churn probability output
    else:
        shap_arr = np.abs(np.array(shap_vals))
    # Ensure 2D: (n_samples, n_features)
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[:, :, -1]  # take churn output dimension
    elif shap_arr.ndim == 1:
        shap_arr = shap_arr.reshape(1, -1)
    mean_shap = shap_arr.mean(axis=0)[:len(feature_names)].tolist()
    shap_importance = sorted(
        [{'feature': fn, 'importance': round(float(v), 4)} for fn, v in zip(feature_names, mean_shap)],
        key=lambda x: x['importance'], reverse=True
    )

    with _model_stats_lock:
        _model_stats_cache.update(dict(
            cm=cm, acc=acc, f1=f1, rec=rec, prec=prec, auc=auc,
            roc_data=roc_data, pr_data=pr_data, shap_importance=shap_importance
        ))
    # Persist to disk so subsequent server restarts load instantly
    try:
        with open(MODEL_STATS_CACHE_FILE, 'w') as _cache_f:
            json.dump(dict(_model_stats_cache), _cache_f)
    except Exception:
        pass


def _try_load_stats_cache() -> bool:
    """Load pre-computed model stats from disk cache. Returns True on success."""
    global _model_stats_cache
    if not os.path.exists(MODEL_STATS_CACHE_FILE):
        return False
    try:
        with open(MODEL_STATS_CACHE_FILE, 'r') as _cache_f:
            data = json.load(_cache_f)
        with _model_stats_lock:
            _model_stats_cache.update(data)
        return True
    except Exception:
        return False


# Load from disk cache first; only run background computation if cache is missing
_cache_loaded = _try_load_stats_cache()
if not _cache_loaded:
    _bg_thread = threading.Thread(target=_build_model_stats, daemon=True)
else:
    _bg_thread = threading.Thread(target=lambda: None, daemon=True)
_bg_thread.start()

def get_data_summary():
    df = pd.read_csv("churn.csv")
    summary = {
        'total_customers': len(df),
        'overall_churn_rate': round(df['Exited'].mean() * 100, 2),
        'avg_age': round(df['Age'].mean(), 2),
        'avg_balance': round(df['Balance'].mean(), 2),
        'avg_credit_score': round(df['CreditScore'].mean(), 2),
        'geography_churn': df.groupby('Geography')['Exited'].mean().to_dict(),
        'gender_churn': df.groupby('Gender')['Exited'].mean().to_dict(),
        'active_member_churn': df.groupby('IsActiveMember')['Exited'].mean().to_dict()
    }
    return summary

@app.route('/')
def home():
    df = pd.read_csv("churn.csv")
    stats = {
        'total': len(df),
        'churn_rate': round(df['Exited'].mean() * 100, 1),
        'avg_balance': f"{df['Balance'].mean():,.0f}",
        'avg_score': round(df['CreditScore'].mean(), 0),
    }
    return render_template('home.html', stats=stats)

@app.route('/predict')
def predict_page():
    return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        df = pd.read_csv("churn.csv")
        total_customers = int(len(df))
        churn_rate = round(float(df['Exited'].mean()) * 100, 2)
        geo_churn = df.groupby('Geography')['Exited'].mean()
        riskiest_geo = str(geo_churn.idxmax())
        riskiest_geo_rate = round(float(geo_churn.max()) * 100, 1)
        top_shap_feature = 'Age'
        model_acc: float = 0.0
        model_auc: float = 0.0
        with _model_stats_lock:
            if _model_stats_cache.get('shap_importance'):
                top_shap_feature = _model_stats_cache['shap_importance'][0]['feature']
            model_acc = float(_model_stats_cache.get('acc', 0.0))
            model_auc = float(_model_stats_cache.get('auc', 0.0))
        return render_template('chat.html',
            total_customers=total_customers,
            churn_rate=churn_rate,
            riskiest_geo=riskiest_geo,
            riskiest_geo_rate=riskiest_geo_rate,
            top_shap_feature=top_shap_feature,
            model_acc=model_acc,
            model_auc=model_auc
        )

    try:
        req_data = request.json
        user_msg: str = req_data.get('message', '')
        history: List[Dict[str, Any]] = req_data.get('history', [])
        summary = get_data_summary()

        shap_summary = ''
        f1_val: float = 0.0
        acc_val: float = 0.0
        auc_val: float = 0.0
        with _model_stats_lock:
            if _model_stats_cache.get('shap_importance'):
                top5 = _model_stats_cache['shap_importance'][:5]
                shap_summary = ', '.join(
                    f"{x['feature']} ({x['importance']:.4f})" for x in top5
                )
            f1_val  = float(_model_stats_cache.get('f1',  0))
            acc_val = float(_model_stats_cache.get('acc', 0))
            auc_val = float(_model_stats_cache.get('auc', 0))

        system_prompt = f"""You are ChurnGuard AI, an expert banking intelligence analyst embedded in the ChurnGuard retention platform.

Dataset Overview:
- Total Customers: {summary['total_customers']:,}
- Overall Churn Rate: {summary['overall_churn_rate']}%
- Average Age: {summary['avg_age']}
- Average Balance: ${summary['avg_balance']:,.2f}
- Average Credit Score: {summary['avg_credit_score']}

Churn by Geography: {summary['geography_churn']}
Churn by Gender (Female=0, Male=1): {summary['gender_churn']}
Churn by Active Membership (1=Active, 0=Inactive): {summary['active_member_churn']}

ANN Model Performance:
- Accuracy: {acc_val}%  |  ROC-AUC: {auc_val}%  |  F1 Score: {f1_val}%

Top 5 SHAP Feature Importances (global mean |SHAP|):
{shap_summary if shap_summary else 'Age, Balance, IsActiveMember, NumOfProducts, Geography'}

Key Dataset Findings:
- Germany has the highest churn rate (~32%) among all geographies
- Inactive members are approximately 3x more likely to churn than active members
- Customers with 3-4 products have churn rates approaching 80%
- Age is the strongest single predictor of churn in the SHAP analysis

Customer Segments (K-Means, 4 Personas):
- Dormant Affluents: Older, high-balance, low-engagement customers. Highest churn risk.
- Young Transients: Young, short-tenure customers. Sensitive to product value proposition.
- Loyal Core: Mid-age, long-tenure, strong credit. Most loyal and stable segment.
- High-Value Actives: High-balance, active, premium credit scores. Highest lifetime value.

Response Formatting Rules:
- Use **bold** for key metrics and important terms
- Use bullet points for lists (prefix with "- ")
- Use numbered lists for rankings or steps (prefix with "1. ")
- Keep responses concise, data-rich, and actionable
- If asked about something outside the dataset scope, state limitations clearly
"""

        # Build multi-turn content from client-side history
        contents: List[types.Content] = []
        for h in history:
            role = 'user' if h.get('role') == 'user' else 'model'
            contents.append(types.Content(role=role, parts=[types.Part(text=h.get('text', ''))]))
        contents.append(types.Content(role='user', parts=[types.Part(text=user_msg)]))

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            config=types.GenerateContentConfig(system_instruction=system_prompt),
            contents=contents
        )

        return jsonify({'success': True, 'reply': response.text})
    except Exception as e:
        app.logger.error(f"Chat failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    df = pd.read_csv("churn.csv")

    # KPI Cards
    total_customers = len(df)
    churn_rate = round(df['Exited'].mean() * 100, 2)
    avg_balance = round(df['Balance'].mean(), 2)
    avg_credit_score = round(df['CreditScore'].mean(), 1)
    active_pct = round(df['IsActiveMember'].mean() * 100, 1)

    # Geography churn
    geo_stats = df.groupby('Geography')['Exited'].mean().to_dict()

    # Age distribution
    age_churn = df[df['Exited'] == 1]['Age'].tolist()
    age_stay = df[df['Exited'] == 0]['Age'].tolist()

    # Balance comparison
    balance_stats = {
        'Churned': round(df[df['Exited'] == 1]['Balance'].mean(), 2),
        'Stayed': round(df[df['Exited'] == 0]['Balance'].mean(), 2)
    }

    # Gender churn rate
    gender_churn = df.groupby('Gender')['Exited'].mean().round(4).to_dict()

    # Products churn rate
    products_churn = df.groupby('NumOfProducts')['Exited'].mean().round(4).to_dict()
    products_count = df.groupby('NumOfProducts').size().to_dict()

    # Tenure churn rate
    tenure_churn = df.groupby('Tenure')['Exited'].mean().round(4).to_dict()

    # Active member churn
    active_churn = df.groupby('IsActiveMember')['Exited'].mean().round(4).to_dict()

    # Credit Score bands
    df['CreditBand'] = pd.cut(df['CreditScore'], bins=[300, 580, 669, 739, 799, 850],
                               labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
    credit_churn = df.groupby('CreditBand', observed=True)['Exited'].mean().round(4).to_dict()

    # Geography × Gender churn (richer than salary quintile)
    geo_gender = df.groupby(['Geography', 'Gender'])['Exited'].mean().round(4)
    geos = ['France', 'Germany', 'Spain']
    male_churn = {g: round(float(geo_gender.get((g, 'Male'), 0)) * 100, 1) for g in geos}
    female_churn = {g: round(float(geo_gender.get((g, 'Female'), 0)) * 100, 1) for g in geos}

    return render_template('dashboard.html',
        geo_stats=geo_stats,
        age_churn=age_churn,
        age_stay=age_stay,
        balance_stats=balance_stats,
        total_customers=total_customers,
        churn_rate=churn_rate,
        avg_balance=avg_balance,
        avg_credit_score=avg_credit_score,
        active_pct=active_pct,
        gender_churn=gender_churn,
        products_churn=products_churn,
        products_count=products_count,
        tenure_churn=tenure_churn,
        active_churn=active_churn,
        credit_churn=credit_churn,
        male_churn=male_churn,
        female_churn=female_churn
    )

@app.route('/personas')
def personas():
    df = pd.read_csv("churn.csv")
    cluster_features = ['Age', 'Balance', 'CreditScore', 'EstimatedSalary']
    x_cluster = df[cluster_features]
    x_scaled = cluster_scaler.transform(x_cluster)
    
    df['Cluster'] = kmeans.predict(x_scaled)
    
    persona_data = []
    for i in range(4):
        cluster_df = df[df['Cluster'] == i]
        persona_data.append({
            'id': i,
            'name': PERSONA_NAMES.get(i, f'Cluster {i}'),
            'icon': PERSONA_ICONS.get(i, 'fa-users'),
            'description': PERSONA_DESC.get(i, ''),
            'avg_age': round(cluster_df['Age'].mean(), 1),
            'avg_balance': round(cluster_df['Balance'].mean(), 2),
            'avg_credit': round(cluster_df['CreditScore'].mean(), 1),
            'avg_salary': round(cluster_df['EstimatedSalary'].mean(), 2),
            'avg_tenure': round(cluster_df['Tenure'].mean(), 1),
            'active_pct': round(cluster_df['IsActiveMember'].mean() * 100, 1),
            'churn_rate': round(cluster_df['Exited'].mean() * 100, 2),
            'count': len(cluster_df)
        })
    
    plot_df = df.sample(n=1000, random_state=0)
    plot_data = {
        'x': plot_df['Age'].tolist(),
        'y': plot_df['Balance'].tolist(),
        'z': plot_df['CreditScore'].tolist(),
        'cluster': plot_df['Cluster'].tolist()
    }

    return render_template('personas.html', persona_data=persona_data, plot_data=plot_data,
                           persona_names=PERSONA_NAMES)


@app.route('/segments')
def segments():
    df = pd.read_csv("churn.csv")

    def _agg(group_col: str, label_col: Optional[str] = None) -> List[Dict[str, Any]]:
        g = df.groupby(group_col)
        result = []
        for key, grp in g:
            record: Dict[str, Any] = {
                group_col: key,
                'label': str(key) if label_col is None else ('Active' if key == 1 else 'Inactive'),
                'churn_rate': round(float(grp['Exited'].mean()) * 100, 2),
                'count': int(len(grp)),
                'avg_balance': round(float(grp['Balance'].mean()), 2),
                'avg_age': round(float(grp['Age'].mean()), 1),
            }
            result.append(record)
        return result

    geo_data = _agg('Geography')
    for d in geo_data:
        d['label'] = d['Geography']

    activity_data = _agg('IsActiveMember', label_col='IsActiveMember')

    products_data = _agg('NumOfProducts')
    for d in products_data:
        n = int(d['NumOfProducts'])
        d['label'] = f"{n} Product{'s' if n > 1 else ''}"

    df['CreditBand'] = pd.cut(
        df['CreditScore'],
        bins=[300, 580, 669, 739, 799, 850],
        labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    )
    credit_data_raw = df.groupby('CreditBand', observed=True)
    credit_data: List[Dict[str, Any]] = []
    for key, grp in credit_data_raw:
        credit_data.append({
            'CreditBand': str(key),
            'label': str(key),
            'churn_rate': round(float(grp['Exited'].mean()) * 100, 2),
            'count': int(len(grp)),
            'avg_balance': round(float(grp['Balance'].mean()), 2),
            'avg_age': round(float(grp['Age'].mean()), 1),
        })

    return render_template('segments.html',
        geo_data=geo_data,
        activity_data=activity_data,
        products_data=products_data,
        credit_data=credit_data,
        total=int(len(df)),
        overall_churn=round(float(df['Exited'].mean()) * 100, 2)
    )


@app.route('/segments/playbook', methods=['POST'])
def segments_playbook():
    try:
        data = request.json
        segment_label = data.get('segment_label', '')
        churn_rate = data.get('churn_rate', 0)
        avg_age = data.get('avg_age', 0)
        avg_balance = data.get('avg_balance', 0)
        count = data.get('count', 0)
        dimension = data.get('dimension', '')

        shap_top = ''
        with _model_stats_lock:
            if _model_stats_cache.get('shap_importance'):
                shap_top = ', '.join(x['feature'] for x in _model_stats_cache['shap_importance'][:3])

        prompt = f"""You are ChurnGuard AI, an expert banking retention strategist.

Segment Analysis:
- Segment: {segment_label} (dimension: {dimension})
- Churn Rate: {churn_rate}%
- Customer Count: {count:,}
- Average Age: {avg_age}
- Average Balance: ${avg_balance:,.2f}
- Global Top SHAP Drivers: {shap_top if shap_top else 'Age, Balance, Activity Status'}

Generate a professional retention playbook for this customer segment. Provide exactly 3 targeted, actionable retention strategies.

Format your response EXACTLY as:

**Action 1: [Action Title]**
[2-3 sentences explaining the action, why it works for this specific segment, and the expected outcome.]

**Action 2: [Action Title]**
[2-3 sentences explaining the action, why it works for this specific segment, and the expected outcome.]

**Action 3: [Action Title]**
[2-3 sentences explaining the action, why it works for this specific segment, and the expected outcome.]

Be specific, data-driven, and use the segment metrics to justify each recommendation."""

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        return jsonify({'success': True, 'playbook': response.text})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/lab')
def lab():
    return render_template('lab.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = [
            float(data['CreditScore']),
            geo_map[data['Geography']],
            gender_map[data['Gender']],
            float(data['Age']),
            float(data['Tenure']),
            float(data['Balance']),
            float(data['NumOfProducts']),
            float(data['HasCrCard']),
            float(data['IsActiveMember']),
            float(data['EstimatedSalary'])
        ]
        
        scaled_features = scaler.transform(np.array([features]))
        prediction = model.predict(scaled_features)
        output = float(prediction[0][0])
        
        result = "Churn" if output > 0.5 else "Stay"
        confidence = round(output * 100, 2) if result == "Churn" else round((1 - output) * 100, 2)

        # SHAP
        shap_values = explainer.shap_values(scaled_features, nsamples=100)
        vals = shap_values[0].flatten().tolist() if isinstance(shap_values, list) else shap_values.flatten().tolist()
            
        explanations = sorted([
            {'feature': name, 'influence': round(float(val), 4), 'type': 'positive' if val > 0 else 'negative'}
            for name, val in zip(feature_names, vals)
        ], key=lambda x: abs(float(x['influence'])), reverse=True)

        # Persona / Cluster match
        cluster_features = [float(data['Age']), float(data['Balance']), float(data['CreditScore']), float(data['EstimatedSalary'])]
        cluster_scaled = cluster_scaler.transform(np.array([cluster_features]))
        cluster_id = int(kmeans.predict(cluster_scaled)[0])

        return jsonify({
            'success': True,
            'result': result,
            'confidence': confidence,
            'explanations': explanations,
            'cluster_id': cluster_id,
            'persona_name': PERSONA_NAMES.get(cluster_id, f'Cluster {cluster_id}'),
            'persona_icon': PERSONA_ICONS.get(cluster_id, 'fa-users'),
            'persona_desc': PERSONA_DESC.get(cluster_id, '')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate_strategy', methods=['POST'])
def generate_strategy():
    try:
        data = request.json
        # Construct prompt based on top 3 SHAP factors
        top_factors = [f"{e['feature']} ({e['type']} impact)" for e in data['explanations'][:3]]
        
        prompt = f"""
        System: You are an expert banking retention AI.
        Customer Profile: {data['result']} with {data['confidence']}% confidence.
        Top Risk Factors: {', '.join(top_factors)}.
        Task: Write a short, professional retention email (max 150 words) offering a specific banking benefit to keep this customer.
        Be polite and personal.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        strategy = response.text
        
        return jsonify({'success': True, 'strategy': strategy})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/save_strategy', methods=['POST'])
def save_strategy():
    try:
        data = request.json
        conn = sqlite3.connect('retention.db')
        c = conn.cursor()
        c.execute("INSERT INTO strategies (customer_data, prediction, strategy) VALUES (?, ?, ?)",
                 (str(data['customer_data']), data['prediction'], data['strategy']))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=A4,
                             rightMargin=40, leftMargin=40,
                             topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()

    is_churn = data.get('result') == 'Churn'
    risk_color = colors.HexColor('#ff003c') if is_churn else colors.HexColor('#0aff60')
    dark_bg = colors.HexColor('#0d0e15')
    mid_bg = colors.HexColor('#14141e')

    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    title_style = ParagraphStyle('Title', fontName='Helvetica-Bold', fontSize=22,
                                  textColor=colors.HexColor('#00f3ff'), alignment=TA_CENTER,
                                  spaceAfter=4)
    subtitle_style = ParagraphStyle('Sub', fontName='Helvetica', fontSize=10,
                                     textColor=colors.HexColor('#8b9bb4'), alignment=TA_CENTER,
                                     spaceAfter=20)
    section_style = ParagraphStyle('Section', fontName='Helvetica-Bold', fontSize=13,
                                    textColor=colors.HexColor('#00f3ff'), spaceBefore=18, spaceAfter=8)
    body_style = ParagraphStyle('Body', fontName='Helvetica', fontSize=9,
                                 textColor=colors.HexColor('#d1d5db'), spaceAfter=4, leading=14)

    # Header
    story.append(Paragraph("CHURNGUARD INTELLIGENCE REPORT", title_style))
    story.append(Paragraph("AI-Powered Customer Retention Analysis | March 2026", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#00f3ff'), spaceAfter=16))

    # Risk Banner
    verdict_text = f"VERDICT: {'HIGH CHURN RISK' if is_churn else 'CUSTOMER LIKELY TO STAY'}"
    conf_text = f"Model Confidence: {data.get('confidence', 0)}%  |  Persona: {data.get('persona_name', 'Unknown')}"
    banner_data = [[verdict_text], [conf_text]]
    banner_table = Table(banner_data, colWidths=[doc.width])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), risk_color),
        ('BACKGROUND', (0, 1), (-1, 1), mid_bg),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black if not is_churn else colors.white),
        ('TEXTCOLOR', (0, 1), (-1, 1), colors.HexColor('#8b9bb4')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('FONTSIZE', (0, 1), (-1, 1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('BOX', (0, 0), (-1, -1), 1, risk_color),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 16))

    # Customer Profile
    story.append(Paragraph("CUSTOMER PROFILE", section_style))
    cdata = data.get('customer_data', {})
    profile_rows = [
        ['Field', 'Value', 'Field', 'Value'],
        ['Geography', str(cdata.get('Geography', '-')), 'Gender', str(cdata.get('Gender', '-'))],
        ['Age', str(cdata.get('Age', '-')), 'Tenure', str(cdata.get('Tenure', '-')) + ' yrs'],
        ['Credit Score', str(cdata.get('CreditScore', '-')), 'Balance', f"${float(cdata.get('Balance', 0)):,.2f}"],
        ['Num of Products', str(cdata.get('NumOfProducts', '-')), 'Est. Salary', f"${float(cdata.get('EstimatedSalary', 0)):,.2f}"],
        ['Has Credit Card', 'Yes' if str(cdata.get('HasCrCard', '0')) == '1' else 'No',
         'Active Member', 'Yes' if str(cdata.get('IsActiveMember', '0')) == '1' else 'No'],
    ]
    profile_table = Table(profile_rows, colWidths=[1.4 * inch, 1.6 * inch, 1.4 * inch, 1.6 * inch])
    profile_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00f3ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 1), (-1, -1), mid_bg),
        ('TEXTCOLOR', (0, 1), (0, -1), colors.HexColor('#00f3ff')),
        ('TEXTCOLOR', (2, 1), (2, -1), colors.HexColor('#00f3ff')),
        ('TEXTCOLOR', (1, 1), (1, -1), colors.white),
        ('TEXTCOLOR', (3, 1), (3, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#1a1a2e')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [mid_bg, colors.HexColor('#0a0a14')]),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(profile_table)
    story.append(Spacer(1, 12))

    # SHAP Factors
    story.append(Paragraph("TOP RISK FACTORS (SHAP ANALYSIS)", section_style))
    shap_rows = [['#', 'Feature', 'Impact Direction', 'SHAP Value']]
    for i, exp in enumerate(data.get('explanations', [])[:7], 1):
        direction = 'Increases Churn Risk' if exp['type'] == 'positive' else 'Reduces Churn Risk'
        shap_rows.append([str(i), exp['feature'], direction, f"{abs(exp['influence']):.4f}"])
    shap_table = Table(shap_rows, colWidths=[0.3 * inch, 1.8 * inch, 2.2 * inch, 1.0 * inch])
    shap_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#bc13fe')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 1), (-1, -1), mid_bg),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [mid_bg, colors.HexColor('#0a0a14')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#1a1a2e')),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (3, 0), (3, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(shap_table)

    # Persona
    if data.get('persona_name'):
        story.append(Spacer(1, 12))
        story.append(Paragraph("CUSTOMER PERSONA", section_style))
        persona_text = f"<b>{data['persona_name']}</b> — {data.get('persona_desc', '')}"
        story.append(Paragraph(persona_text, body_style))

    # AI Strategy
    if data.get('strategy'):
        story.append(Spacer(1, 12))
        story.append(Paragraph("AI-GENERATED RETENTION STRATEGY", section_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#bc13fe'), spaceAfter=8))
        for line in data['strategy'].split('\n'):
            if line.strip():
                story.append(Paragraph(line, body_style))

    # Footer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#1a1a2e'), spaceAfter=6))
    footer_style = ParagraphStyle('Footer', fontName='Helvetica', fontSize=7,
                                   textColor=colors.HexColor('#8b9bb4'), alignment=TA_CENTER)
    story.append(Paragraph("Generated by ChurnGuard Intelligence Platform | Confidential | March 2026", footer_style))

    doc.build(story)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="ChurnGuard_Report.pdf", mimetype='application/pdf')


@app.route('/model')
def model_page():
    # Block only if background thread hasn't finished yet
    _bg_thread.join(timeout=120)
    with _model_stats_lock:
        cache = dict(_model_stats_cache)
    if not cache:
        return render_template('model.html', loading=True,
            cm=[[0,0],[0,0]], acc=0, f1=0, rec=0, prec=0, auc=0,
            roc_data={'fpr':[0,1],'tpr':[0,1]}, pr_data={'precision':[1,0],'recall':[0,1]},
            shap_importance=[{'feature':f,'importance':0} for f in feature_names])
    return render_template('model.html', loading=False, **cache)


@app.route('/model/recompute', methods=['POST'])
def model_recompute():
    """Delete disk cache and trigger a fresh background recompute."""
    global _bg_thread, _model_stats_cache
    try:
        if os.path.exists(MODEL_STATS_CACHE_FILE):
            os.remove(MODEL_STATS_CACHE_FILE)
        with _model_stats_lock:
            _model_stats_cache.clear()
        _bg_thread = threading.Thread(target=_build_model_stats, daemon=True)
        _bg_thread.start()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/insights')
def insights():
    df = pd.read_csv("churn.csv")

    # Summary stats
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    stats = df[num_cols].describe().round(2).to_dict()

    # Correlation matrix (numeric only)
    df_enc = df.copy()
    df_enc['Geography'] = df_enc['Geography'].map(geo_map)
    df_enc['Gender'] = df_enc['Gender'].map(gender_map)
    corr_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography', 'Gender', 'Exited']
    corr = df_enc[corr_cols].corr().round(3)
    corr_data = {'labels': corr_cols, 'values': corr.values.tolist()}

    # Distribution data for each numeric feature (histograms as bins)
    dist_data: Dict[str, Any] = {}
    for col in num_cols:
        counts, bin_edges = np.histogram(df[col], bins=20)
        dist_data[col] = {
            'counts': counts.tolist(),
            'bins': [round(float(b), 1) for b in bin_edges[:-1]]
        }

    # Sample data (first 100 rows)
    sample = df.head(100).to_dict(orient='records')

    # Business insight cards
    germany_churn = round(df[df['Geography'] == 'Germany']['Exited'].mean() * 100, 1)
    age45plus_churn = round(df[df['Age'] >= 45]['Exited'].mean() * 100, 1)
    inactive_churn = round(df[df['IsActiveMember'] == 0]['Exited'].mean() * 100, 1)
    single_prod_churn = round(df[df['NumOfProducts'] == 1]['Exited'].mean() * 100, 1)
    multi_prod_churn = round(df[df['NumOfProducts'] >= 3]['Exited'].mean() * 100, 1)
    zero_bal_churn = round(df[df['Balance'] == 0]['Exited'].mean() * 100, 1)

    insights_cards = [
        {'icon': 'fa-map-marker-alt', 'color': '#bc13fe', 'title': 'Germany High-Risk',
         'value': f'{germany_churn}%', 'desc': 'Churn rate among German customers — highest of all regions.'},
        {'icon': 'fa-user-clock', 'color': '#ff003c', 'title': 'Age 45+ Risk',
         'value': f'{age45plus_churn}%', 'desc': 'Churn rate for customers aged 45 and above.'},
        {'icon': 'fa-moon', 'color': '#ffe600', 'title': 'Inactive Member Risk',
         'value': f'{inactive_churn}%', 'desc': 'Inactive members churn at a significantly higher rate.'},
        {'icon': 'fa-box', 'color': '#00f3ff', 'title': 'Single Product Churn',
         'value': f'{single_prod_churn}%', 'desc': 'Customers with only 1 product are at elevated churn risk.'},
        {'icon': 'fa-exclamation-triangle', 'color': '#ff003c', 'title': '3+ Products Danger',
         'value': f'{multi_prod_churn}%', 'desc': 'Customers with 3 or more products show extreme churn rates.'},
        {'icon': 'fa-wallet', 'color': '#0aff60', 'title': 'Zero Balance Churn',
         'value': f'{zero_bal_churn}%', 'desc': 'Customers with zero balance churn at a lower rate — dormant but retained.'},
    ]

    return render_template('insights.html',
        stats=stats, corr_data=corr_data, dist_data=dist_data,
        sample=sample, insights_cards=insights_cards, num_cols=num_cols
    )


@app.route('/history')
def history():
    conn = sqlite3.connect('retention.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM strategies ORDER BY timestamp DESC")
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    churn_count = sum(1 for r in rows if r['prediction'] == 'Churn')
    stay_count = len(rows) - churn_count
    return render_template('history.html', rows=rows, churn_count=churn_count, stay_count=stay_count)


@app.route('/delete_strategy/<int:strategy_id>', methods=['DELETE'])
def delete_strategy(strategy_id: int):
    try:
        conn = sqlite3.connect('retention.db')
        c = conn.cursor()
        c.execute("DELETE FROM strategies WHERE id = ?", (strategy_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'success': False, 'error': 'No file uploaded'})

        df = pd.read_csv(file)
        required_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                         'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return jsonify({'success': False, 'error': f'Missing columns: {missing}'})

        df_proc = df.copy()
        df_proc['Geography'] = df_proc['Geography'].map(geo_map).fillna(0)
        df_proc['Gender'] = df_proc['Gender'].map(gender_map).fillna(0)
        X = df_proc[required_cols].values.astype(float)
        X_scaled = scaler.transform(X)
        probs = model.predict(X_scaled).flatten()

        df['Churn_Probability'] = np.round(probs * 100, 2)
        df['Prediction'] = np.where(probs > 0.5, 'Churn', 'Stay')
        cluster_feats = df_proc[['Age', 'Balance', 'CreditScore', 'EstimatedSalary']].values.astype(float)
        cluster_scaled = cluster_scaler.transform(cluster_feats)
        cluster_ids = kmeans.predict(cluster_scaled)
        df['Persona'] = [PERSONA_NAMES.get(int(c), f'Cluster {c}') for c in cluster_ids]

        out = io.StringIO()
        df.to_csv(out, index=False)
        out.seek(0)
        return send_file(io.BytesIO(out.getvalue().encode()),
                         as_attachment=True,
                         download_name='ChurnGuard_Batch_Results.csv',
                         mimetype='text/csv')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0")
