"""
api_consumer.py — Customer Churn Prediction MLOps
Author: Suresh D R | AI Product Developer & Technology Mentor | DV Analytics

This file shows HOW other systems (CRM, billing, call centre)
can consume the Churn Prediction REST API.

Use cases:
  1. CRM system sends customer data → gets churn probability → updates customer record
  2. Billing system detects bill increase → checks churn risk → triggers retention
  3. Call centre tool loads daily high-risk list → agents work through it
  4. Batch scoring → score all customers daily → store in database

Run: python src/api_consumer.py
"""

import requests
import json
import pandas as pd
from datetime import datetime

# ── API Configuration ──────────────────────────────────────────
# Change this to your deployed API URL
# Local testing:  API_URL = "http://localhost:8000"
# Production EKS: API_URL = "http://EXTERNAL-IP:8000"
API_URL = "http://localhost:8000"

def check_api_health():
    """Check if the API is running and model is loaded."""
    print("Checking API health...")
    response = requests.get(f"{API_URL}/health", timeout=10)
    health   = response.json()
    print(f"  Status       : {health.get('status')}")
    print(f"  Model        : {health.get('model_name')}")
    print(f"  AUC-ROC      : {health.get('model_auc')}")
    print(f"  Threshold    : {health.get('threshold')}")
    return health.get('status') == 'healthy'

def predict_single_customer(customer_data):
    """
    Use Case 1 — CRM System:
    Send one customer's data and get churn prediction back.
    """
    print("\nPredicting churn for single customer...")
    response = requests.post(
        f"{API_URL}/predict",
        json=customer_data,
        timeout=30
    )
    if response.status_code == 200:
        result = response.json()
        print(f"  Churn Probability : {result['churn_probability']:.1%}")
        print(f"  Prediction        : {result['churn_label']}")
        print(f"  Risk Tier         : {result['risk_tier']}")
        print(f"  Retention Action  : {result['retention_action']}")
        if result.get('shap_explanation'):
            print(f"  Top Churn Reasons :")
            for feat, val in list(result['shap_explanation'].items())[:3]:
                direction = "↑ increases churn" if val > 0 else "↓ decreases churn"
                print(f"    {feat.replace('_',' '):<35} {direction}")
        return result
    else:
        print(f"  Error: {response.status_code} — {response.text}")
        return None

def predict_batch_customers(customer_list):
    """
    Use Case 2 — Batch Scoring:
    Score all customers at once.
    Call centres use this to get daily high-risk customer list.
    """
    print(f"\nBatch scoring {len(customer_list)} customers...")
    response = requests.post(
        f"{API_URL}/predict/batch",
        json=customer_list,
        timeout=60
    )
    if response.status_code == 200:
        result     = response.json()
        total      = result['count']
        high_risk  = result['high_risk_count']
        print(f"  Total scored     : {total}")
        print(f"  High risk (≥50%) : {high_risk}")
        print(f"  Low risk (<50%)  : {total - high_risk}")
        return result['predictions']
    else:
        print(f"  Error: {response.status_code} — {response.text}")
        return []

def score_from_csv(csv_path):
    """
    Use Case 3 — Billing System Integration:
    Read customers from CSV, score them, add churn probability column.
    Used by billing teams to flag customers before bill increase.
    """
    print(f"\nScoring customers from CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df):,} customers")

        # Score in batches of 50
        results_list = []
        batch_size   = 50

        for i in range(0, min(len(df), 500), batch_size):
            batch   = df.iloc[i:i+batch_size]
            records = []

            for _, row in batch.iterrows():
                records.append({
                    'contract_type'       : str(row.get('contract_type', 'Month-to-Month')),
                    'tenure_months'       : int(row.get('tenure_months', 6)),
                    'monthly_charges'     : float(row.get('monthly_charges', 599)),
                    'total_charges'       : float(row.get('total_charges', 3594)),
                    'num_complaints'      : int(row.get('num_complaints', 0)),
                    'csat_score'          : float(row.get('csat_score', 7.0)),
                    'bill_increase_3m'    : float(row.get('bill_increase_3m', 0.0)),
                    'discount_used'       : int(row.get('discount_used', 0)),
                    'discount_expiry_days': int(row.get('discount_expiry_days', 999)),
                    'days_to_renewal'     : int(row.get('days_to_renewal', 30)),
                    'competitor_offer'    : int(row.get('competitor_offer', 0)),
                    'payment_delays'      : int(row.get('payment_delays', 0)),
                    'complaint_resolution': str(row.get('complaint_resolution', 'No Complaints')),
                    'avg_data_usage_gb'   : float(row.get('avg_data_usage_gb', 10.0)),
                    'last_30d_data_gb'    : float(row.get('last_30d_data_gb', 8.0)),
                    'engagement_index'    : float(row.get('engagement_index', 0.5)),
                    'city_tier'           : str(row.get('city_tier', 'Metro')),
                    'income_segment'      : str(row.get('income_segment', 'Mid')),
                    'payment_method'      : str(row.get('payment_method', 'Auto-Pay')),
                    'referral_source'     : str(row.get('referral_source', 'Organic')),
                    'num_products'        : int(row.get('num_products', 2)),
                    'usage_trend_3m'      : float(row.get('usage_trend_3m', 0.0)),
                    'days_since_last_login': int(row.get('days_since_last_login', 5)),
                    'num_support_calls'   : int(row.get('num_support_calls', 0)),
                    'refund_requested'    : int(row.get('refund_requested', 0)),
                    'feature_adoption_rate': float(row.get('feature_adoption_rate', 0.4)),
                    'tech_savviness'      : str(row.get('tech_savviness', 'Medium')),
                })

            try:
                preds = predict_batch_customers(records)
                for pred in preds:
                    results_list.append({
                        'churn_probability': pred['churn_probability'],
                        'risk_tier'        : pred['risk_tier'],
                        'churn_prediction' : pred['churn_prediction'],
                        'retention_action' : pred['retention_action'],
                    })
            except Exception as e:
                print(f"  Batch {i//batch_size+1} failed: {e}")
                for _ in records:
                    results_list.append({
                        'churn_probability': None,
                        'risk_tier': 'Unknown',
                        'churn_prediction': None,
                        'retention_action': 'Manual review',
                    })

        if results_list:
            results_df = pd.DataFrame(results_list)
            output_df  = df.head(len(results_df)).copy()
            output_df['churn_probability'] = results_df['churn_probability'].values
            output_df['risk_tier']         = results_df['risk_tier'].values
            output_df['retention_action']  = results_df['retention_action'].values
            output_df['scored_at']         = datetime.now().strftime('%Y-%m-%d %H:%M')

            # Sort by churn risk
            output_df = output_df.sort_values('churn_probability', ascending=False)

            output_path = csv_path.replace('.csv', '_scored.csv')
            output_df.to_csv(output_path, index=False)
            print(f"  Scored file saved: {output_path}")

            # Summary
            risk_counts = output_df['risk_tier'].value_counts()
            print(f"\n  Risk Tier Summary:")
            for tier, count in risk_counts.items():
                print(f"    {tier:<15}: {count:,} customers")

            return output_df
    except FileNotFoundError:
        print(f"  CSV file not found: {csv_path}")
        return None

def simulate_crm_integration():
    """
    Use Case 4 — Simulated CRM Integration:
    Shows how a real CRM would call the API for each customer interaction.
    """
    print("\n" + "="*60)
    print("SIMULATED CRM INTEGRATION")
    print("="*60)
    print("Scenario: Customer service agent opens a customer record.")
    print("CRM automatically calls the churn API and shows risk score.")
    print()

    customers = [
        {
            "name": "Rajesh Kumar",
            "data": {
                "contract_type": "Month-to-Month", "tenure_months": 3,
                "monthly_charges": 699, "num_complaints": 5,
                "csat_score": 2.8, "bill_increase_3m": 0.22,
                "discount_used": 1, "discount_expiry_days": 8,
                "days_to_renewal": 4, "competitor_offer": 1,
                "complaint_resolution": "Escalated", "payment_delays": 3,
                "engagement_index": 0.15, "avg_data_usage_gb": 10,
                "last_30d_data_gb": 3, "income_segment": "Mid",
            }
        },
        {
            "name": "Priya Sharma",
            "data": {
                "contract_type": "Two Year", "tenure_months": 36,
                "monthly_charges": 399, "num_complaints": 0,
                "csat_score": 9.2, "bill_increase_3m": 0.0,
                "discount_used": 0, "days_to_renewal": 180,
                "competitor_offer": 0, "complaint_resolution": "No Complaints",
                "payment_delays": 0, "engagement_index": 0.85,
                "avg_data_usage_gb": 15, "last_30d_data_gb": 14,
                "income_segment": "High",
            }
        },
        {
            "name": "Mohammed Ali",
            "data": {
                "contract_type": "Month-to-Month", "tenure_months": 8,
                "monthly_charges": 549, "num_complaints": 2,
                "csat_score": 6.0, "bill_increase_3m": 0.12,
                "discount_used": 1, "discount_expiry_days": 25,
                "days_to_renewal": 12, "competitor_offer": 1,
                "complaint_resolution": "Pending", "payment_delays": 1,
                "engagement_index": 0.40, "avg_data_usage_gb": 12,
                "last_30d_data_gb": 8, "income_segment": "Mid",
            }
        },
    ]

    all_results = []
    for customer in customers:
        print(f"\nCustomer: {customer['name']}")
        print("-" * 40)
        result = predict_single_customer(customer['data'])
        if result:
            all_results.append({
                'Customer'   : customer['name'],
                'Risk'       : result['risk_tier'],
                'Probability': f"{result['churn_probability']:.1%}",
                'Action'     : result['retention_action'][:50]
            })

    if all_results:
        print("\n" + "="*60)
        print("CRM DASHBOARD — CUSTOMER RISK SUMMARY")
        print("="*60)
        summary_df = pd.DataFrame(all_results).sort_values('Probability', ascending=False)
        print(summary_df.to_string(index=False))

def main():
    print("="*60)
    print("CUSTOMER CHURN API CONSUMER — DEMO")
    print(f"API URL: {API_URL}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 1. Health check
    if not check_api_health():
        print("\nAPI is not healthy! Start the API first:")
        print("  python src/api.py")
        return

    # 2. Single customer prediction
    high_risk_customer = {
        "contract_type"       : "Month-to-Month",
        "tenure_months"       : 3,
        "monthly_charges"     : 699.0,
        "total_charges"       : 2097.0,
        "num_complaints"      : 5,
        "csat_score"          : 2.8,
        "bill_increase_3m"    : 0.22,
        "discount_used"       : 1,
        "discount_expiry_days": 8,
        "days_to_renewal"     : 4,
        "competitor_offer"    : 1,
        "complaint_resolution": "Escalated",
        "payment_delays"      : 3,
        "engagement_index"    : 0.15,
        "avg_data_usage_gb"   : 10.0,
        "last_30d_data_gb"    : 3.0,
        "income_segment"      : "Mid",
        "city_tier"           : "Metro",
        "payment_method"      : "Manual",
        "referral_source"     : "Promotion",
        "num_products"        : 1,
        "usage_trend_3m"      : -0.40,
        "days_since_last_login": 18,
        "num_support_calls"   : 8,
        "refund_requested"    : 1,
        "feature_adoption_rate": 0.15,
        "tech_savviness"      : "Low",
    }

    print("\n" + "="*60)
    print("USE CASE 1 — SINGLE CUSTOMER PREDICTION")
    print("="*60)
    predict_single_customer(high_risk_customer)

    # 3. Batch prediction
    print("\n" + "="*60)
    print("USE CASE 2 — BATCH PREDICTION (5 customers)")
    print("="*60)
    batch_customers = [high_risk_customer] * 5
    batch_results   = predict_batch_customers(batch_customers)
    if batch_results:
        print(f"  First result — Churn prob: {batch_results[0]['churn_probability']:.1%}")

    # 4. CSV scoring
    print("\n" + "="*60)
    print("USE CASE 3 — SCORE FROM CSV")
    print("="*60)
    print("  Usage: score_from_csv('data/current/new_policies.csv')")
    print("  Saves scored file with churn_probability column added")

    # 5. CRM integration simulation
    simulate_crm_integration()

    print("\n" + "="*60)
    print("API CONSUMER DEMO COMPLETE")
    print("="*60)
    print()
    print("To use in your system:")
    print(f"  API Base URL  : {API_URL}")
    print(f"  Predict single: POST {API_URL}/predict")
    print(f"  Batch score   : POST {API_URL}/predict/batch")
    print(f"  Health check  : GET  {API_URL}/health")
    print(f"  Model info    : GET  {API_URL}/model_info")
    print()
    print("API Documentation: http://localhost:8000/docs")

if __name__ == '__main__':
    main()
