"""
api.py — Customer Churn Prediction MLOps
Author: Suresh D R | AI Product Developer & Technology Mentor | DV Analytics

FastAPI REST endpoint for system integration.
CRM systems, call centres, billing systems can call this API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import predict, get_model_info

app = FastAPI(
    title="Customer Churn Prediction API",
    description="""
    Predicts customer churn probability for retention team prioritisation.
    Author: Suresh D R | AI Product Developer & Technology Mentor | DV Analytics

    ## Endpoints
    - **POST /predict** — Predict churn for a single customer
    - **POST /predict/batch** — Predict churn for multiple customers
    - **GET /health** — API and model health check
    - **GET /model_info** — Current model version and metrics
    """,
    version="1.0.0"
)

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Request Schema ─────────────────────────────────────────────
class CustomerRecord(BaseModel):
    # Contract
    contract_type        : str   = Field(default="Month-to-Month")
    plan_type            : str   = Field(default="Monthly")
    tenure_months        : int   = Field(default=6)
    payment_method       : str   = Field(default="Manual")
    monthly_charges      : float = Field(default=599.0)
    total_charges        : float = Field(default=3594.0)
    num_products         : int   = Field(default=1)
    paperless_billing    : int   = Field(default=1)
    bill_increase_3m     : float = Field(default=0.05)
    days_to_renewal      : int   = Field(default=30)
    # Usage
    avg_data_usage_gb    : float = Field(default=10.0)
    last_30d_data_gb     : float = Field(default=7.0)
    avg_monthly_calls    : int   = Field(default=100)
    last_30d_calls       : int   = Field(default=80)
    days_since_last_login: int   = Field(default=5)
    usage_trend_3m       : float = Field(default=-0.10)
    engagement_index     : float = Field(default=0.50)
    feature_adoption_rate: float = Field(default=0.40)
    # Service
    num_complaints       : int   = Field(default=1)
    complaint_resolution : str   = Field(default="Resolved")
    csat_score           : float = Field(default=7.0)
    nps_score            : int   = Field(default=20)
    num_support_calls    : int   = Field(default=2)
    avg_resolution_days  : float = Field(default=3.0)
    # Billing
    payment_delays       : int   = Field(default=0)
    discount_used        : int   = Field(default=0)
    discount_expiry_days : int   = Field(default=999)
    refund_requested     : int   = Field(default=0)
    credit_card_expiry   : int   = Field(default=0)
    # Profile
    customer_age         : int   = Field(default=35)
    city_tier            : str   = Field(default="Metro")
    income_segment       : str   = Field(default="Mid")
    tech_savviness       : str   = Field(default="Medium")
    referral_source      : str   = Field(default="Organic")
    competitor_offer     : int   = Field(default=0)
    # Other
    roaming_usage        : int   = Field(default=0)
    international_calls  : int   = Field(default=0)
    num_devices          : int   = Field(default=2)
    avg_session_duration : float = Field(default=25.0)
    agent_quality_score  : float = Field(default=7.0)
    last_complaint_days  : int   = Field(default=30)
    last_payment_days    : int   = Field(default=15)
    disputed_bills       : int   = Field(default=0)
    market_tenure_years  : float = Field(default=5.0)

    class Config:
        json_schema_extra = {
            "example": {
                "contract_type": "Month-to-Month",
                "tenure_months": 3,
                "monthly_charges": 599,
                "num_complaints": 4,
                "csat_score": 3.2,
                "bill_increase_3m": 0.18,
                "discount_used": 1,
                "discount_expiry_days": 10,
                "competitor_offer": 1
            }
        }

# ── Endpoints ──────────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        info = get_model_info()
        return {
            "status"      : "healthy",
            "model_loaded": True,
            "model_name"  : info.get("model", "Unknown"),
            "model_auc"   : info.get("auc", "N/A"),
            "threshold"   : info.get("threshold", "N/A"),
            "api_version" : "1.0.0"
        }
    except Exception as e:
        return {"status": "degraded", "model_loaded": False, "error": str(e)}

@app.get("/model_info")
def model_info():
    try:
        return get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_churn(record: CustomerRecord):
    """
    Predict churn probability for a single customer.
    Returns: churn_probability, risk_tier, retention_action, SHAP explanation.
    """
    try:
        # Add derived features
        d = record.dict()
        d['usage_decline_score'] = (d['last_30d_data_gb'] - d['avg_data_usage_gb']) / (d['avg_data_usage_gb'] + 0.01)
        d['complaint_intensity'] = d['num_complaints'] / (d['tenure_months'] + 0.01)
        d['value_score']         = d['total_charges'] / (d['tenure_months'] + 0.01)
        result = predict(d)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
def predict_batch(records: list[CustomerRecord]):
    """Predict churn for multiple customers at once."""
    try:
        results = []
        for r in records:
            d = r.dict()
            d['usage_decline_score'] = (d['last_30d_data_gb'] - d['avg_data_usage_gb']) / (d['avg_data_usage_gb'] + 0.01)
            d['complaint_intensity'] = d['num_complaints'] / (d['tenure_months'] + 0.01)
            d['value_score']         = d['total_charges'] / (d['tenure_months'] + 0.01)
            results.append(predict(d))
        return {"predictions": results, "count": len(results),
                "high_risk_count": sum(1 for r in results if r['churn_probability'] >= 0.5)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch failed: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
