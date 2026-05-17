"""
predict.py — Customer Churn Prediction MLOps
Author: Suresh D R | AI Product Developer & Technology Mentor | DV Analytics

Core prediction module. Loads best model from S3 and returns churn predictions.
Used by: app.py, api.py
"""

import numpy as np
import pandas as pd
import boto3
import io
import os
import json
import joblib
import warnings
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings('ignore')

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID',     'YOUR_AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'YOUR_AWS_SECRET_KEY')
BUCKET         = os.getenv('S3_BUCKET',             'customer-churn-project-2024')
REGION         = os.getenv('AWS_REGION',            'ap-south-1')

# ── Model cache ────────────────────────────────────────────────
_model         = None
_feature_names = None
_target_means  = None
_threshold     = None
_model_metrics = None

def get_s3():
    return boto3.client('s3', region_name=REGION,
                        aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY)

def load_model():
    global _model, _feature_names, _target_means, _threshold, _model_metrics
    if _model is not None:
        return _model, _feature_names, _target_means, _threshold, _model_metrics

    s3 = get_s3()
    print("Loading model from S3...")

    obj    = s3.get_object(Bucket=BUCKET, Key='models/best_model.pkl')
    _model = joblib.load(io.BytesIO(obj['Body'].read()))

    try:
        obj            = s3.get_object(Bucket=BUCKET, Key='models/feature_names.pkl')
        _feature_names = joblib.load(io.BytesIO(obj['Body'].read()))
    except: _feature_names = None

    try:
        obj           = s3.get_object(Bucket=BUCKET, Key='models/target_means.pkl')
        _target_means = joblib.load(io.BytesIO(obj['Body'].read()))
    except: _target_means = {}

    try:
        obj        = s3.get_object(Bucket=BUCKET, Key='models/threshold.pkl')
        thresh_obj = joblib.load(io.BytesIO(obj['Body'].read()))
        _threshold = thresh_obj.get('threshold', 0.4)
    except: _threshold = 0.4

    try:
        obj            = s3.get_object(Bucket=BUCKET, Key='models/model_metrics.json')
        _model_metrics = json.loads(obj['Body'].read())
    except: _model_metrics = {}

    print(f"Model loaded! AUC={_model_metrics.get('auc','N/A')} "
          f"Threshold={_threshold:.2f}")
    return _model, _feature_names, _target_means, _threshold, _model_metrics

# ── Encoding maps ─────────────────────────────────────────────
BINARY_COLS  = ['paperless_billing', 'roaming_usage', 'international_calls',
                'discount_used', 'refund_requested', 'credit_card_expiry',
                'competitor_offer']
ORDINAL_MAPS = {
    'city_tier'     : {'Metro': 2, 'Tier2': 1, 'Tier3': 0},
    'income_segment': {'High': 2, 'Mid': 1, 'Low': 0},
    'tech_savviness': {'High': 2, 'Medium': 1, 'Low': 0},
}
TARGET_ENCODE_COLS = ['contract_type', 'payment_method', 'referral_source',
                      'complaint_resolution', 'plan_type']

def get_risk_tier(prob):
    if prob >= 0.70:   return "Critical",  "IMMEDIATE retention call — senior agent assigned"
    elif prob >= 0.50: return "High",      "Automated personalised offer via SMS and app"
    elif prob >= 0.30: return "Medium",    "Loyalty points top-up — monitor closely"
    else:              return "Low",       "No action required — monitor monthly"

def preprocess_input(record_dict, feature_names, target_means):
    df = pd.DataFrame([record_dict])

    for col in ['contract_type', 'payment_method', 'city_tier',
                'income_segment', 'complaint_resolution', 'referral_source']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Engineer features
    if 'monthly_charges' in df.columns and 'tenure_months' in df.columns:
        df['revenue_at_risk'] = df['monthly_charges'] * (
            12 - float(df['tenure_months'].values[0].clip(0, 12) if hasattr(
                df['tenure_months'].values[0], 'clip') else min(max(df['tenure_months'].values[0], 0), 12)))
    if 'num_complaints' in df.columns and 'bill_increase_3m' in df.columns:
        df['complaint_bill_risk'] = df['num_complaints'] * (
            1 + np.clip(df['bill_increase_3m'], 0, 1))
    if 'days_to_renewal' in df.columns:
        df['churn_window_flag'] = (df['days_to_renewal'] < 14).astype(int)
    if 'tenure_months' in df.columns and 'num_products' in df.columns:
        df['loyalty_score'] = (df['tenure_months'] / 12) * df['num_products']
    if 'num_complaints' in df.columns and 'csat_score' in df.columns:
        df['service_dissatisfaction'] = (
            df['num_complaints'] * 0.4 +
            (10 - df['csat_score'].fillna(5)) * 0.3 +
            df.get('payment_delays', pd.Series([0])).fillna(0) * 0.3)
    if 'engagement_index' in df.columns and 'feature_adoption_rate' in df.columns:
        df['digital_engagement'] = df['engagement_index'] * df['feature_adoption_rate']

    # Encode
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    for col in TARGET_ENCODE_COLS:
        if col in df.columns and target_means and col in target_means:
            global_mean = np.mean(list(target_means[col].values()))
            df[col]     = df[col].map(target_means[col]).fillna(global_mean)

    for col in df.select_dtypes(include='object').columns:
        df[col] = 0

    df = df.fillna(0)

    if feature_names:
        for f in feature_names:
            if f not in df.columns:
                df[f] = 0
        df = df[[f for f in feature_names if f in df.columns]]

    return df

def predict(record_dict):
    """
    Main prediction function.
    Returns churn probability, risk tier, SHAP explanation, retention action.
    """
    model, feature_names, target_means, threshold, model_metrics = load_model()

    X = preprocess_input(record_dict, feature_names, target_means)

    churn_prob  = float(model.predict_proba(X)[0][1])
    churn_pred  = int(churn_prob >= threshold)
    risk_tier, action = get_risk_tier(churn_prob)

    # SHAP
    shap_explanation = {}
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        vals      = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        feat_imp  = pd.DataFrame({'feature': feature_names or X.columns,
                                   'shap': vals}
                                 ).sort_values('shap', key=abs, ascending=False).head(6)
        shap_explanation = {
            row['feature']: round(float(row['shap']), 4)
            for _, row in feat_imp.iterrows()
        }
    except: pass

    return {
        'churn_probability' : round(churn_prob, 4),
        'churn_prediction'  : churn_pred,
        'churn_label'       : "WILL CHURN" if churn_pred else "WILL STAY",
        'risk_tier'         : risk_tier,
        'retention_action'  : action,
        'threshold_used'    : threshold,
        'shap_explanation'  : shap_explanation,
        'model_info'        : {
            'model_name'  : model_metrics.get('model', 'Unknown'),
            'auc'         : model_metrics.get('auc', 0),
            'trained_on'  : model_metrics.get('trained_on', 'Unknown'),
        }
    }

def get_model_info():
    _, _, _, _, metrics = load_model()
    return metrics

if __name__ == '__main__':
    sample = {
        'tenure_months': 3, 'contract_type': 'Month-to-Month',
        'monthly_charges': 599, 'num_complaints': 4,
        'csat_score': 3.2, 'bill_increase_3m': 0.18,
        'discount_used': 1, 'discount_expiry_days': 10,
        'days_to_renewal': 5, 'engagement_index': 0.25,
        'usage_decline_score': -0.40, 'num_products': 1,
        'payment_method': 'Manual', 'city_tier': 'Metro',
        'competitor_offer': 1, 'payment_delays': 2,
        'complaint_resolution': 'Escalated', 'referral_source': 'Promotion',
        'income_segment': 'Mid', 'feature_adoption_rate': 0.20,
    }
    result = predict(sample)
    print("\nPrediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")
