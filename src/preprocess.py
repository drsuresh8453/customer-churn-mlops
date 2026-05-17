"""
preprocess.py — Customer Churn Prediction MLOps
Author: Suresh D R | AI Product Developer & Technology Mentor | DV Analytics

Handles all data cleaning, imputation, encoding and feature selection.
Used by: train.py, predict.py, SageMaker Pipeline
"""

import pandas as pd
import numpy as np
import boto3
import io
import os
import joblib
import warnings
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings('ignore')

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID',     'YOUR_AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'YOUR_AWS_SECRET_KEY')
BUCKET         = os.getenv('S3_BUCKET',             'customer-churn-project-2024')
REGION         = os.getenv('AWS_REGION',            'ap-south-1')

def get_s3():
    return boto3.client('s3', region_name=REGION,
                        aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY)

def read_csv_s3(key):
    s3  = get_s3()
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    df  = pd.read_csv(io.BytesIO(obj['Body'].read()))
    print(f"Loaded: {key} — {df.shape[0]:,} rows")
    return df

def save_csv_s3(df, key):
    s3  = get_s3()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    print(f"Saved: s3://{BUCKET}/{key}")

# ── Encoding maps ─────────────────────────────────────────────
BINARY_COLS = ['paperless_billing', 'roaming_usage', 'international_calls',
               'discount_used', 'refund_requested', 'credit_card_expiry',
               'competitor_offer', 'churn_window_flag']

ORDINAL_MAPS = {
    'city_tier'     : {'Metro': 2, 'Tier2': 1, 'Tier3': 0},
    'income_segment': {'High': 2, 'Mid': 1, 'Low': 0},
    'tech_savviness': {'High': 2, 'Medium': 1, 'Low': 0},
}

TARGET_ENCODE_COLS = ['contract_type', 'payment_method', 'referral_source',
                      'complaint_resolution', 'plan_type']

def clean_data(df):
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicates removed: {before - len(df)}")

    num_cols = ['tenure_months', 'monthly_charges', 'total_charges',
                'avg_monthly_calls', 'csat_score', 'nps_score']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'tenure_months' in df.columns:
        df = df[df['tenure_months'] >= 0]
    if 'monthly_charges' in df.columns:
        df = df[df['monthly_charges'] >= 0]
    if 'customer_age' in df.columns:
        df = df[(df['customer_age'] >= 18) & (df['customer_age'] <= 90)]

    for col in ['contract_type', 'payment_method', 'city_tier',
                'income_segment', 'complaint_resolution', 'referral_source']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    print(f"  Clean dataset: {len(df):,} rows")
    return df

def impute_missing(df):
    df = df.copy()
    if 'csat_score' in df.columns:
        df['csat_score'] = df.groupby('complaint_resolution')['csat_score'].transform(
            lambda x: x.fillna(x.median()))
        df['csat_score'] = df['csat_score'].fillna(df['csat_score'].median())

    if 'nps_score' in df.columns:
        df['nps_score'] = df.groupby('contract_type')['nps_score'].transform(
            lambda x: x.fillna(x.median()))
        df['nps_score'] = df['nps_score'].fillna(df['nps_score'].median())

    for col in ['avg_session_duration', 'agent_quality_score', 'market_tenure_years']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    print(f"  Missing values remaining: {df.isnull().sum().sum()}")
    return df

def cap_outliers(df):
    df = df.copy()
    cap_cols = ['monthly_charges', 'total_charges', 'avg_data_usage_gb',
                'num_support_calls', 'avg_session_duration']
    for col in cap_cols:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR    = Q3 - Q1
            df[col] = df[col].clip(lower=max(Q1 - 1.5*IQR, 0), upper=Q3 + 1.5*IQR)
    print(f"  Outliers capped")
    return df

def engineer_features(df):
    df = df.copy()
    if 'monthly_charges' in df.columns and 'tenure_months' in df.columns:
        df['revenue_at_risk'] = df['monthly_charges'] * (
            12 - df['tenure_months'].clip(0, 12))
    if 'num_complaints' in df.columns and 'bill_increase_3m' in df.columns:
        df['complaint_bill_risk'] = df['num_complaints'] * (
            1 + df['bill_increase_3m'].clip(0, 1))
    if 'days_to_renewal' in df.columns:
        df['churn_window_flag'] = (df['days_to_renewal'] < 14).astype(int)
    if 'tenure_months' in df.columns and 'num_products' in df.columns:
        df['loyalty_score'] = (df['tenure_months'] / 12) * df['num_products']
    if 'num_complaints' in df.columns and 'csat_score' in df.columns:
        df['service_dissatisfaction'] = (
            df['num_complaints'] * 0.4 +
            (10 - df['csat_score'].fillna(5)) * 0.3 +
            df['payment_delays'].fillna(0) * 0.3)
    if 'engagement_index' in df.columns and 'feature_adoption_rate' in df.columns:
        df['digital_engagement'] = (
            df['engagement_index'] * df['feature_adoption_rate'])
    return df

def encode_features(df, target_means=None, fit=True):
    df = df.copy()

    # Binary cols → int
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Ordinal encoding
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    # Target encoding
    if fit:
        target_means = {}
        for col in TARGET_ENCODE_COLS:
            if col in df.columns and 'churn' in df.columns:
                mean_map = df.groupby(col)['churn'].mean().to_dict()
                target_means[col] = mean_map
                df[col] = df[col].map(mean_map).fillna(df['churn'].mean())
    else:
        for col in TARGET_ENCODE_COLS:
            if col in df.columns and target_means and col in target_means:
                global_mean = np.mean(list(target_means[col].values()))
                df[col]     = df[col].map(target_means[col]).fillna(global_mean)

    # Remaining object columns → int codes
    for col in df.select_dtypes(include='object').columns:
        if col not in ['churn']:
            df[col] = pd.Categorical(df[col]).codes

    df = df.fillna(0)
    df = df.select_dtypes(include=[np.number])
    print(f"  Encoded: {df.shape[1]} features")
    return df, target_means

def run_preprocessing_pipeline(source='s3', save_to_s3=True):
    print("=" * 55)
    print("PREPROCESSING PIPELINE")
    print("=" * 55)

    if source == 's3':
        df = read_csv_s3('raw/customer_churn_raw.csv')
    else:
        df = pd.read_csv(source)

    print("\nStep 1: Cleaning...")
    df = clean_data(df)
    print("\nStep 2: Imputing...")
    df = impute_missing(df)
    print("\nStep 3: Capping outliers...")
    df = cap_outliers(df)
    print("\nStep 4: Engineering features...")
    df = engineer_features(df)
    print("\nStep 5: Encoding...")
    df_encoded, target_means = encode_features(df, fit=True)

    if save_to_s3:
        save_csv_s3(df_encoded, 'data/06_encoded_tree.csv')
        s3  = get_s3()
        buf = io.BytesIO()
        joblib.dump(target_means, buf); buf.seek(0)
        s3.put_object(Bucket=BUCKET, Key='models/target_means.pkl', Body=buf.getvalue())

    print(f"\nPreprocessing complete! Shape: {df_encoded.shape}")
    return df_encoded, target_means

def preprocess_single(record_dict, target_means=None):
    """Preprocess a single customer record for prediction."""
    df = pd.DataFrame([record_dict])

    for col in ['contract_type', 'payment_method', 'city_tier',
                'income_segment', 'complaint_resolution', 'referral_source']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    df = engineer_features(df)

    # Load target means from S3 if not provided
    if target_means is None:
        try:
            s3  = get_s3()
            obj = s3.get_object(Bucket=BUCKET, Key='models/target_means.pkl')
            target_means = joblib.load(io.BytesIO(obj['Body'].read()))
        except Exception as e:
            print(f"Warning: Could not load target means: {e}")
            target_means = {}

    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    for col in TARGET_ENCODE_COLS:
        if col in df.columns and col in target_means:
            global_mean = np.mean(list(target_means[col].values()))
            df[col]     = df[col].map(target_means[col]).fillna(global_mean)

    for col in df.select_dtypes(include='object').columns:
        df[col] = 0

    df = df.fillna(0)

    # Align to training features
    try:
        s3       = get_s3()
        obj      = s3.get_object(Bucket=BUCKET, Key='models/feature_names.pkl')
        feat_names = joblib.load(io.BytesIO(obj['Body'].read()))
        for f in feat_names:
            if f not in df.columns:
                df[f] = 0
        df = df[[f for f in feat_names if f in df.columns]]
    except Exception as e:
        print(f"Warning: Feature alignment: {e}")
        df = df.select_dtypes(include=[np.number])

    return df

if __name__ == '__main__':
    run_preprocessing_pipeline()
