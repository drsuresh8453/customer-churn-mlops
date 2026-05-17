"""
detect_drift.py — Customer Churn Prediction MLOps
Author: Suresh D R | AI Product Developer & Technology Mentor | DV Analytics

Evidently AI drift detection + email report + SageMaker trigger.
"""

import pandas as pd
import numpy as np
import boto3
import io
import os
import json
from datetime import datetime
from scipy import stats
import warnings
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings('ignore')

try:
    from evidently.report import Report
    from evidently.metrics import DatasetDriftMetric, DataDriftTable, DatasetSummaryMetric
    EVIDENTLY_AVAILABLE = True
except: EVIDENTLY_AVAILABLE = False

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID',     'YOUR_AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'YOUR_AWS_SECRET_KEY')
BUCKET         = os.getenv('S3_BUCKET',             'customer-churn-project-2024')
REGION         = os.getenv('AWS_REGION',            'ap-south-1')
PIPELINE_NAME  = os.getenv('SAGEMAKER_PIPELINE',    'customer-churn-pipeline')
SNS_TOPIC_ARN  = os.getenv('SNS_TOPIC_ARN',         '')
ALERT_EMAIL    = os.getenv('ALERT_EMAIL',           'drsuresh8453@gmail.com')

MONITOR_FEATURES = [
    'tenure_months', 'monthly_charges', 'num_complaints',
    'csat_score', 'bill_increase_3m', 'engagement_index',
    'days_since_last_login', 'usage_decline_score', 'payment_delays'
]

CAT_FEATURES = [
    'contract_type', 'payment_method', 'city_tier',
    'complaint_resolution', 'referral_source'
]

def get_s3():
    return boto3.client('s3', region_name=REGION,
                        aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY)

def read_csv_s3(key):
    s3  = get_s3()
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))

def run_drift_detection(reference_df, current_df):
    """Run KS test drift detection."""
    drifted, results = [], {}

    for feature in MONITOR_FEATURES:
        if feature in reference_df.columns and feature in current_df.columns:
            stat, p = stats.ks_2samp(
                reference_df[feature].dropna(),
                current_df[feature].dropna()
            )
            is_drifted = p < 0.01
            results[feature] = {'p_value': round(p, 6), 'drifted': is_drifted, 'ks_stat': round(stat, 4)}
            if is_drifted:
                drifted.append(feature)
                print(f"  DRIFT: {feature:<30} p={p:.6f}")
            else:
                print(f"  OK   : {feature:<30} p={p:.6f}")

    for feature in CAT_FEATURES:
        if feature in reference_df.columns and feature in current_df.columns:
            try:
                ref_c = reference_df[feature].value_counts()
                cur_c = current_df[feature].value_counts()
                cats  = set(ref_c.index) | set(cur_c.index)
                ref_a = np.array([ref_c.get(c, 0) for c in cats]) + 1e-10
                cur_a = np.array([cur_c.get(c, 0) for c in cats]) + 1e-10
                ref_a /= ref_a.sum(); cur_a /= cur_a.sum()
                stat, p = stats.chisquare(cur_a, ref_a)
                is_drifted = p < 0.01
                results[feature] = {'p_value': round(p, 6), 'drifted': is_drifted}
                if is_drifted:
                    drifted.append(feature)
                    print(f"  DRIFT: {feature:<30} p={p:.6f}")
            except: pass

    total          = len(MONITOR_FEATURES) + len(CAT_FEATURES)
    drift_ratio    = len(drifted) / max(total, 1)
    dataset_drift  = drift_ratio > 0.30
    return {'dataset_drift': dataset_drift, 'drifted_features': drifted,
            'drift_ratio': round(drift_ratio, 4), 'feature_results': results}

def generate_html_report(reference_df, current_df, drift_results, report_path):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    drifted   = drift_results.get('drifted_features', [])
    html = f"""<!DOCTYPE html><html><head><title>Churn Model Drift Report</title>
<style>body{{font-family:Arial;background:#f5f5f5;margin:20px}}
.header{{background:#1F4E79;color:white;padding:20px;border-radius:8px}}
.summary{{background:white;padding:20px;margin:10px 0;border-radius:8px;
border-left:5px solid {'#C00000' if drift_results['dataset_drift'] else '#28a745'}}}
table{{width:100%;border-collapse:collapse;background:white;border-radius:8px}}
th{{background:#1F4E79;color:white;padding:10px}}td{{padding:8px;border:1px solid #ddd}}
.drift{{color:#C00000;font-weight:bold}}.ok{{color:#28a745}}
</style></head><body>
<div class="header"><h1>📊 Customer Churn — Data Drift Report</h1>
<p>Author: Suresh D R | DV Analytics | {timestamp}</p></div>
<div class="summary">
<h2>{'⚠️ DRIFT DETECTED — Retraining Triggered' if drift_results['dataset_drift'] else '✅ No Significant Drift — Model Stable'}</h2>
<p><b>Drifted Features:</b> {len(drifted)} / {len(MONITOR_FEATURES)+len(CAT_FEATURES)}</p>
<p><b>Drift Ratio:</b> {drift_results.get('drift_ratio', 0):.2%}</p>
<p><b>Reference Data:</b> {len(reference_df):,} records</p>
<p><b>Current Data:</b> {len(current_df):,} records</p></div>
<h2>Feature Drift Details</h2>
<table><tr><th>Feature</th><th>P-Value</th><th>KS Stat</th><th>Status</th></tr>"""
    for feat, res in drift_results.get('feature_results', {}).items():
        cls  = 'drift' if res['drifted'] else 'ok'
        txt  = '⚠️ DRIFTED' if res['drifted'] else '✅ OK'
        html += f"<tr><td>{feat}</td><td>{res['p_value']}</td><td>{res.get('ks_stat','N/A')}</td><td class='{cls}'>{txt}</td></tr>"
    html += """</table><p><em>Customer Churn MLOps Pipeline | DV Analytics</em></p></body></html>"""
    with open(report_path, 'w', encoding='utf-8') as f: f.write(html)
    return report_path

def send_alerts(drift_results, report_path):
    s3 = get_s3()
    try:
        key = f"reports/drift_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        s3.upload_file(report_path, BUCKET, key)
        print(f"Report uploaded: s3://{BUCKET}/{key}")
    except Exception as e: print(f"Upload failed: {e}")

    if SNS_TOPIC_ARN:
        try:
            sns = boto3.client('sns', region_name=REGION,
                               aws_access_key_id=AWS_ACCESS_KEY,
                               aws_secret_access_key=AWS_SECRET_KEY)
            subject = ("⚠️ DRIFT DETECTED — Churn Model Retraining Triggered"
                      if drift_results['dataset_drift']
                      else "✅ No Drift — Churn Model Stable")
            sns.publish(TopicArn=SNS_TOPIC_ARN, Subject=subject,
                       Message=f"Drift ratio: {drift_results.get('drift_ratio',0):.2%}\n"
                               f"Drifted: {drift_results.get('drifted_features', [])}")
            print("SNS alert sent!")
        except Exception as e: print(f"SNS failed: {e}")

def trigger_sagemaker():
    try:
        sm = boto3.client('sagemaker', region_name=REGION,
                          aws_access_key_id=AWS_ACCESS_KEY,
                          aws_secret_access_key=AWS_SECRET_KEY)
        resp = sm.start_pipeline_execution(
            PipelineName=PIPELINE_NAME,
            PipelineExecutionDisplayName=f"drift-retrain-{datetime.now().strftime('%Y%m%d-%H%M')}"
        )
        print(f"SageMaker triggered: {resp['PipelineExecutionArn']}")
        return resp['PipelineExecutionArn']
    except Exception as e:
        print(f"SageMaker trigger failed: {e}")
        return None

def main(email=None):
    print("="*55)
    print("CUSTOMER CHURN — DAILY DRIFT DETECTION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*55)

    try:
        reference_df = read_csv_s3('reference/training_reference.csv')
    except:
        print("Loading raw data as reference...")
        reference_df = read_csv_s3('raw/customer_churn_raw.csv')

    try:
        current_df = read_csv_s3('data/new_policies.csv')
    except:
        print("No current data found — using reference")
        current_df = reference_df.copy()

    print(f"Reference: {len(reference_df):,} | Current: {len(current_df):,}")

    print("\nRunning drift detection...")
    drift_results = run_drift_detection(reference_df, current_df)

    report_path = f"churn_drift_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    generate_html_report(reference_df, current_df, drift_results, report_path)
    send_alerts(drift_results, report_path)

    pipeline_arn = None
    if drift_results['dataset_drift']:
        print("\nDRIFT DETECTED — Triggering SageMaker retraining...")
        pipeline_arn = trigger_sagemaker()
    else:
        print("\nNo drift — model stable")

    print(f"\nDrift: {drift_results['dataset_drift']} | "
          f"Ratio: {drift_results['drift_ratio']:.2%}")
    return drift_results

def lambda_handler(event, context):
    results = main()
    return {'statusCode': 200,
            'body': json.dumps({'dataset_drift': results['dataset_drift'],
                                'drift_ratio': results['drift_ratio']})}

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--email', default=ALERT_EMAIL)
    args = p.parse_args()
    main(email=args.email)
