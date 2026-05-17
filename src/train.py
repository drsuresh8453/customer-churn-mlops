"""
train.py — Customer Churn Prediction MLOps
Author: Suresh D R | AI Product Developer & Technology Mentor | DV Analytics

Trains best churn model on encoded data from S3.
Evaluates against existing model — saves if better.
Logs all experiments to MLflow.
"""

import pandas as pd
import numpy as np
import boto3
import io
import os
import json
import joblib
import warnings
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, accuracy_score)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except: XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except: LGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CB_AVAILABLE = True
except: CB_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except: MLFLOW_AVAILABLE = False

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID',     'YOUR_AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'YOUR_AWS_SECRET_KEY')
BUCKET         = os.getenv('S3_BUCKET',             'customer-churn-project-2024')
REGION         = os.getenv('AWS_REGION',            'ap-south-1')
MLFLOW_URI     = f's3://{BUCKET}/mlflow'

def get_s3():
    return boto3.client('s3', region_name=REGION,
                        aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY)

def read_csv_s3(key):
    s3  = get_s3()
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))

def save_model_s3(model, key):
    s3  = get_s3()
    buf = io.BytesIO()
    joblib.dump(model, buf); buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    print(f"Saved: s3://{BUCKET}/{key}")

def evaluate_model(name, y_true, y_pred, y_prob, dataset='Test'):
    auc  = roc_auc_score(y_true, y_prob)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    acc  = accuracy_score(y_true, y_pred)
    print(f"  {dataset:6} — AUC:{auc:.4f} F1:{f1:.4f} "
          f"Prec:{prec:.4f} Rec:{rec:.4f} Acc:{acc:.4f}")
    return {'Model': name, 'AUC': round(auc,4), 'F1': round(f1,4),
            'Precision': round(prec,4), 'Recall': round(rec,4)}

def cv_evaluate(model, X, y, cv=2):
    skf  = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    aucs = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    f1s  = cross_val_score(model, X, y, cv=skf, scoring='f1',      n_jobs=-1)
    print(f"  {cv}-Fold CV:")
    for i, (a, f) in enumerate(zip(aucs, f1s), 1):
        print(f"    Fold {i}: AUC={a:.4f}  F1={f:.4f}")
    print(f"    Mean : AUC={aucs.mean():.4f}  F1={f1s.mean():.4f}")
    return aucs.mean(), f1s.mean()

def find_optimal_threshold(y_true, y_prob,
                            cost_missed=2500, cost_false_alarm=200):
    """Find threshold that minimises business cost."""
    best_t, best_cost = 0.5, float('inf')
    for t in np.arange(0.1, 0.9, 0.02):
        pred = (y_prob >= t).astype(int)
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        cost = fn * cost_missed + fp * cost_false_alarm
        if cost < best_cost:
            best_cost = cost
            best_t    = t
    return best_t, best_cost

def train_models(X_train, X_test, y_train, y_test):
    results, models = [], {}
    scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()

    # XGBoost
    if XGB_AVAILABLE:
        print("\nTraining XGBoost...")
        xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            scale_pos_weight=scale_pos_weight,
            random_state=42, verbosity=0, tree_method='hist'
        )
        xgb.fit(X_train, y_train)
        metrics = evaluate_model("XGBoost", y_test,
                                  xgb.predict(X_test),
                                  xgb.predict_proba(X_test)[:,1])
        cv_auc, cv_f1 = cv_evaluate(xgb, X_train, y_train, cv=2)
        metrics['cv_auc'] = round(cv_auc, 4)
        results.append(metrics); models['XGBoost'] = xgb

    # LightGBM
    if LGB_AVAILABLE:
        print("\nTraining LightGBM...")
        lgbm = lgb.LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.08,
            num_leaves=50, subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, verbose=-1
        )
        lgbm.fit(X_train, y_train)
        metrics = evaluate_model("LightGBM", y_test,
                                  lgbm.predict(X_test),
                                  lgbm.predict_proba(X_test)[:,1])
        cv_auc, cv_f1 = cv_evaluate(lgbm, X_train, y_train, cv=2)
        metrics['cv_auc'] = round(cv_auc, 4)
        results.append(metrics); models['LightGBM'] = lgbm

    # CatBoost
    if CB_AVAILABLE:
        print("\nTraining CatBoost...")
        cb = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.08,
            class_weights={0: 1, 1: int(scale_pos_weight)},
            random_seed=42, verbose=0, early_stopping_rounds=30
        )
        cb.fit(X_train, y_train, eval_set=(X_test, y_test))
        metrics = evaluate_model("CatBoost", y_test,
                                  cb.predict(X_test),
                                  cb.predict_proba(X_test)[:,1])
        metrics['cv_auc'] = metrics['AUC']
        results.append(metrics); models['CatBoost'] = cb

    return results, models

def main():
    print("=" * 55)
    print("CUSTOMER CHURN — MODEL TRAINING PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # Load data
    print("\nLoading encoded data from S3...")
    df = read_csv_s3('data/06_encoded_tree.csv')
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

    X = df.drop(columns=['churn'])
    y = df['churn']
    print(f"Churn rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train models
    results, models = train_models(X_train, X_test, y_train, y_test)

    # Pick best
    results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
    best_name  = results_df.iloc[0]['Model']
    best_model = models[best_name]
    best_auc   = results_df.iloc[0]['AUC']
    best_prob  = best_model.predict_proba(X_test)[:,1]

    print(f"\nBest model: {best_name} (AUC={best_auc:.4f})")

    # Compare with existing
    is_better = True
    try:
        s3  = get_s3()
        obj = s3.get_object(Bucket=BUCKET, Key='models/model_metrics.json')
        existing = json.loads(obj['Body'].read())
        existing_auc = existing.get('auc', 0)
        print(f"Existing AUC: {existing_auc:.4f} | New AUC: {best_auc:.4f}")
        is_better = best_auc > existing_auc
    except:
        print("No existing model — saving as first version")

    if is_better:
        # Find optimal threshold
        best_t, best_cost = find_optimal_threshold(y_test, best_prob)
        print(f"Optimal threshold: {best_t:.2f} (cost=₹{best_cost:,.0f})")

        # Save model
        save_model_s3(best_model, 'models/best_model.pkl')

        # Save feature names
        s3  = get_s3()
        buf = io.BytesIO()
        joblib.dump(X.columns.tolist(), buf); buf.seek(0)
        s3.put_object(Bucket=BUCKET, Key='models/feature_names.pkl', Body=buf.getvalue())

        # Save threshold
        buf2 = io.BytesIO()
        joblib.dump({'threshold': best_t, 'threshold_f1': best_t}, buf2); buf2.seek(0)
        s3.put_object(Bucket=BUCKET, Key='models/threshold.pkl', Body=buf2.getvalue())

        # Save metrics
        metrics = {
            'model'     : best_name,
            'auc'       : float(best_auc),
            'f1'        : float(results_df.iloc[0]['F1']),
            'precision' : float(results_df.iloc[0]['Precision']),
            'recall'    : float(results_df.iloc[0]['Recall']),
            'threshold' : float(best_t),
            'trained_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_train'   : len(X_train),
            'churn_rate': float(y.mean()),
        }
        s3.put_object(Bucket=BUCKET, Key='models/model_metrics.json',
                      Body=json.dumps(metrics, indent=2))

        print(f"\nNew model saved! AUC={best_auc:.4f}")
    else:
        print("Existing model is better — keeping current model")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return is_better, best_auc

if __name__ == '__main__':
    main()
# CI/CD test
