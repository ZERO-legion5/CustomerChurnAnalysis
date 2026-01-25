from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import shap
import io
import sqlite3
import os
from google import genai
from google.genai import types
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
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
    return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        return render_template('chat.html')
    
    try:
        user_msg = request.json.get('message')
        summary = get_data_summary()
        
        system_prompt = f"""
        You are ChurnGuard AI, a helpful banking intelligence assistant.
        Use these dataset statistics to answer questions:
        - Total Customers: {summary['total_customers']}
        - Overall Churn Rate: {summary['overall_churn_rate']}%
        - Average Age: {summary['avg_age']}
        - Average Balance: ${summary['avg_balance']}
        - Average Credit Score: {summary['avg_credit_score']}
        - Churn by Geography: {summary['geography_churn']}
        - Churn by Gender: {summary['gender_churn']}
        - Churn by Active Membership (1=Active, 0=Inactive): {summary['active_member_churn']}
        
        Be professional, data-driven, and concise. If you don't know something based on the data, say so.
        """
        
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            config=types.GenerateContentConfig(
                system_instruction=system_prompt
            ),
            contents=user_msg
        )
        
        return jsonify({'success': True, 'reply': response.text})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/dashboard')
def dashboard():
    df = pd.read_csv("churn.csv")
    geo_stats = df.groupby('Geography')['Exited'].mean().to_dict()
    age_churn = df[df['Exited'] == 1]['Age'].tolist()
    age_stay = df[df['Exited'] == 0]['Age'].tolist()
    balance_stats = {
        'Churned': df[df['Exited'] == 1]['Balance'].mean(),
        'Stayed': df[df['Exited'] == 0]['Balance'].mean()
    }
    return render_template('dashboard.html', geo_stats=geo_stats, age_churn=age_churn, age_stay=age_stay, balance_stats=balance_stats)

@app.route('/personas')
def personas():
    df = pd.read_csv("churn.csv")
    # Prepare data for clustering (live view)
    cluster_features = ['Age', 'Balance', 'CreditScore', 'EstimatedSalary']
    x_cluster = df[cluster_features]
    x_scaled = cluster_scaler.transform(x_cluster)
    
    df['Cluster'] = kmeans.predict(x_scaled)
    
    # Generate persona names based on cluster centroids
    # This is a bit manual but adds great academic flavor
    persona_data = []
    for i in range(4):
        cluster_df = df[df['Cluster'] == i]
        persona_data.append({
            'id': i,
            'avg_age': round(cluster_df['Age'].mean(), 1),
            'avg_balance': round(cluster_df['Balance'].mean(), 2),
            'avg_credit': round(cluster_df['CreditScore'].mean(), 1),
            'churn_rate': round(cluster_df['Exited'].mean() * 100, 2),
            'count': len(cluster_df)
        })
    
    # Prepare samples for 3D plot
    plot_df = df.sample(n=1000) # Sample 1000 for performance
    plot_data = {
        'x': plot_df['Age'].tolist(),
        'y': plot_df['Balance'].tolist(),
        'z': plot_df['CreditScore'].tolist(),
        'cluster': plot_df['Cluster'].tolist()
    }

    return render_template('personas.html', persona_data=persona_data, plot_data=plot_data)

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
            'cluster_id': cluster_id
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
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 20)
    p.drawString(100, 750, "ChurnGuard Intelligence Report")
    p.setFont("Helvetica", 12)
    p.drawString(100, 720, f"Customer Status: {data['result']}")
    p.drawString(100, 700, f"Model Confidence: {data['confidence']}%")
    p.line(100, 680, 500, 680)
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, 650, "Top Risk Factors:")
    y = 630
    p.setFont("Helvetica", 10)
    for exp in data['explanations'][:5]:
        impact = "Increases Risk" if exp['type'] == 'positive' else "Decreases Risk"
        p.drawString(120, y, f"- {exp['feature']}: {impact} ({exp['influence']})")
        y -= 20
    if 'strategy' in data:
        p.setFont("Helvetica-Bold", 14)
        p.drawString(100, y - 20, "AI Retention Strategy:")
        y -= 40
        p.setFont("Helvetica", 9)
        textobject = p.beginText(100, y)
        for line in data['strategy'].split('\n'):
            textobject.textLine(line)
        p.drawText(textobject)
    p.showPage()
    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="Retention_Analysis.pdf", mimetype='application/pdf')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
