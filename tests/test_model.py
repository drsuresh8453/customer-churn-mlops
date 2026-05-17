"""
test_model.py — Customer Churn Prediction MLOps
Author: Suresh D R | AI Product Developer & Technology Mentor | DV Analytics
8 automated tests — all must pass before deployment.
"""

import pytest
import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

SAMPLE = {
    'contract_type': 'Month-to-Month', 'tenure_months': 3,
    'monthly_charges': 599.0, 'total_charges': 1797.0,
    'num_complaints': 4, 'csat_score': 3.2,
    'bill_increase_3m': 0.18, 'discount_used': 1,
    'discount_expiry_days': 10, 'days_to_renewal': 5,
    'competitor_offer': 1, 'payment_delays': 2,
    'complaint_resolution': 'Escalated', 'referral_source': 'Promotion',
    'income_segment': 'Mid', 'city_tier': 'Metro',
    'payment_method': 'Manual', 'engagement_index': 0.20,
    'avg_data_usage_gb': 10.0, 'last_30d_data_gb': 3.0,
    'feature_adoption_rate': 0.15, 'num_products': 1,
    'usage_trend_3m': -0.40, 'days_since_last_login': 18,
    'num_support_calls': 6, 'refund_requested': 1,
    'tech_savviness': 'Low', 'plan_type': 'Monthly',
}

def test_model_exists_in_s3():
    """Test 1: Model file exists in S3."""
    import boto3
    s3 = boto3.client('s3',
                       region_name=os.getenv('AWS_REGION','ap-south-1'),
                       aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID','YOUR_AWS_ACCESS_KEY'),
                       aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY','YOUR_AWS_SECRET_KEY'))
    try:
        s3.head_object(Bucket='customer-churn-project-2024', Key='models/best_model.pkl')
        print("\nTest 1 PASSED: Model exists in S3")
    except Exception as e:
        pytest.fail(f"Model not found in S3: {e}")

def test_model_loads():
    """Test 2: Model loads without error."""
    from predict import load_model
    try:
        model, feat, means, thresh, metrics = load_model()
        assert model is not None
        print("\nTest 2 PASSED: Model loads successfully")
    except Exception as e:
        pytest.fail(f"Model failed to load: {e}")

def test_prediction_returns_probability():
    """Test 3: Prediction returns a probability."""
    from predict import predict
    result = predict(SAMPLE)
    assert isinstance(result['churn_probability'], (int, float))
    assert not np.isnan(result['churn_probability'])
    print(f"\nTest 3 PASSED: Probability = {result['churn_probability']:.4f}")

def test_probability_valid_range():
    """Test 4: Probability is between 0 and 1."""
    from predict import predict
    result = predict(SAMPLE)
    assert 0 <= result['churn_probability'] <= 1
    print(f"\nTest 4 PASSED: Probability in valid range [0,1]")

def test_risk_tier_valid():
    """Test 5: Risk tier is one of expected values."""
    from predict import predict
    result = predict(SAMPLE)
    assert result['risk_tier'] in ['Critical', 'High', 'Medium', 'Low']
    print(f"\nTest 5 PASSED: Risk tier = {result['risk_tier']}")

def test_churn_label_valid():
    """Test 6: Churn label is valid."""
    from predict import predict
    result = predict(SAMPLE)
    assert result['churn_label'] in ['WILL CHURN', 'WILL STAY']
    print(f"\nTest 6 PASSED: Churn label = {result['churn_label']}")

def test_high_risk_predicts_higher_than_low_risk():
    """Test 7: High-risk customer scores higher than low-risk."""
    from predict import predict

    high_risk = SAMPLE.copy()
    high_risk.update({'contract_type': 'Month-to-Month', 'num_complaints': 8,
                       'csat_score': 2.0, 'bill_increase_3m': 0.30,
                       'discount_expiry_days': 5, 'tenure_months': 2})

    low_risk = SAMPLE.copy()
    low_risk.update({'contract_type': 'Two Year', 'num_complaints': 0,
                      'csat_score': 9.5, 'bill_increase_3m': 0.0,
                      'discount_expiry_days': 999, 'tenure_months': 48})

    high_result = predict(high_risk)
    low_result  = predict(low_risk)

    assert high_result['churn_probability'] > low_result['churn_probability'], \
        f"High risk ({high_result['churn_probability']:.4f}) should be > " \
        f"Low risk ({low_result['churn_probability']:.4f})"
    print(f"\nTest 7 PASSED: "
          f"High={high_result['churn_probability']:.4f} > "
          f"Low={low_result['churn_probability']:.4f}")

def test_model_auc_above_threshold():
    """Test 8: Model AUC is above 0.70."""
    from predict import get_model_info
    info = get_model_info()
    auc  = info.get('auc', 0)
    assert auc > 0.70, f"Model AUC {auc} is below 0.70 threshold"
    print(f"\nTest 8 PASSED: AUC = {auc} (above 0.70 threshold)")

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
