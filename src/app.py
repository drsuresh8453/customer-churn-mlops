"""
app.py — Customer Churn Prediction MLOps
Author: Suresh D R | AI Product Developer & Technology Mentor | DV Analytics

Streamlit web application for retention team.
Shows churn probability, risk tier, SHAP explanation,
and LLM-generated personalised retention message using OpenAI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import predict, get_model_info

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .main-header { font-size:2rem; font-weight:bold; color:#1F4E79;
                 text-align:center; padding:1rem 0; }
  .risk-critical { background:#fff0f0; border-left:5px solid #C00000;
                   padding:12px; border-radius:6px; }
  .risk-high     { background:#fff8f0; border-left:5px solid #FF5722;
                   padding:12px; border-radius:6px; }
  .risk-medium   { background:#fffde7; border-left:5px solid #FFC107;
                   padding:12px; border-radius:6px; }
  .risk-low      { background:#f0fff0; border-left:5px solid #2e7d32;
                   padding:12px; border-radius:6px; }
  .llm-box      { background:#f0f4f8; border:1px solid #1F4E79;
                   padding:16px; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────
st.markdown('<div class="main-header">📊 Customer Churn Predictor</div>',
            unsafe_allow_html=True)
st.markdown("**DV Analytics | Author: Suresh D R | AI Product Developer & Technology Mentor**")
st.markdown("---")

# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # OpenAI API Key
    st.subheader("🤖 LLM Retention Message")
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Enter your OpenAI API key to generate personalised retention messages"
    )
    if openai_key:
        st.success("OpenAI key loaded!")
    else:
        st.info("Add OpenAI key to get AI-generated retention messages")

    st.markdown("---")
    st.subheader("📊 Model Information")
    try:
        info = get_model_info()
        st.success(f"Model: **{info.get('model','N/A')}**")
        st.metric("AUC-ROC",    f"{info.get('auc','N/A')}")
        st.metric("F1 Score",   f"{info.get('f1','N/A')}")
        st.metric("Threshold",  f"{info.get('threshold','N/A')}")
        st.caption(f"Trained: {info.get('trained_on','N/A')}")
    except Exception as e:
        st.warning(f"Model info unavailable: {e}")

    st.markdown("---")
    st.markdown("""
    ### Risk Tiers
    - 🔴 **Critical** → ≥ 70% probability
    - 🟠 **High** → 50-70% probability
    - 🟡 **Medium** → 30-50% probability
    - 🟢 **Low** → < 30% probability
    """)

# ── LLM Retention Message ──────────────────────────────────────
def generate_retention_message(customer_data, prediction_result, openai_key):
    """Generate personalised retention message using OpenAI."""
    if not openai_key:
        return None

    try:
        import openai
        client = openai.OpenAI(api_key=openai_key)

        risk_tier   = prediction_result['risk_tier']
        churn_prob  = prediction_result['churn_probability']
        shap_exp    = prediction_result.get('shap_explanation', {})

        # Build context from SHAP
        top_reasons = []
        for feat, val in list(shap_exp.items())[:3]:
            if val > 0:
                top_reasons.append(feat.replace('_',' '))

        reasons_text = ', '.join(top_reasons) if top_reasons else 'usage decline and service issues'

        prompt = f"""
You are a retention specialist at a telecom/subscription company.
A customer has been flagged as {risk_tier} churn risk with {churn_prob:.0%} probability.

Customer profile:
- Contract type: {customer_data.get('contract_type', 'Unknown')}
- Tenure: {customer_data.get('tenure_months', 0)} months
- Monthly charges: ₹{customer_data.get('monthly_charges', 0):,.0f}
- Number of complaints: {customer_data.get('num_complaints', 0)}
- CSAT score: {customer_data.get('csat_score', 'N/A')}
- Bill increase last 3 months: {customer_data.get('bill_increase_3m', 0):.0%}
- Days to renewal: {customer_data.get('days_to_renewal', 'N/A')}
- Discount expiring: {"Yes — in " + str(customer_data.get('discount_expiry_days','')) + " days" if customer_data.get('discount_used') else "No"}
- Competitor offer available: {"Yes" if customer_data.get('competitor_offer') else "No"}

Top churn drivers: {reasons_text}

Write a warm, personalised retention message (2-3 sentences) that:
1. Acknowledges their specific situation
2. Offers a concrete retention incentive
3. Gives them a clear next step

Keep it friendly, not salesy. Make it feel personal.
Return ONLY the message text, no labels.
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content":
                 "You are a customer retention specialist. Write warm, personalised messages."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    except ImportError:
        return "Install openai: pip install openai"
    except Exception as e:
        return f"LLM error: {str(e)[:100]}"

# ── Input Form ──────────────────────────────────────────────────
st.header("📝 Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Account & Contract")
    contract_type     = st.selectbox("Contract Type",
        ['Month-to-Month', 'One Year', 'Two Year'])
    plan_type         = st.selectbox("Plan Type",
        ['Monthly', 'Semi-Annual', 'Annual', 'Prepaid'])
    tenure_months     = st.slider("Tenure (months)", 1, 120, 6)
    payment_method    = st.selectbox("Payment Method",
        ['Manual', 'Auto-Pay', 'Credit Card', 'Net Banking'])
    monthly_charges   = st.number_input("Monthly Charges (₹)", 100, 10000, 599, 50)
    num_products      = st.selectbox("Number of Products", [1, 2, 3, 4, 5])
    paperless_billing = st.selectbox("Paperless Billing",
        [1, 0], format_func=lambda x: "Yes" if x else "No")
    bill_increase_3m  = st.slider("Bill Increase Last 3 Months (%)", -10, 40, 5) / 100
    days_to_renewal   = st.slider("Days to Renewal", 1, 365, 30)

with col2:
    st.subheader("📞 Usage & Service")
    avg_data_usage_gb    = st.slider("Avg Monthly Data (GB)", 0.0, 50.0, 10.0, 0.5)
    last_30d_data_gb     = st.slider("Last 30 Days Data (GB)", 0.0, 50.0, 7.0, 0.5)
    days_since_last_login = st.slider("Days Since Last Login", 0, 60, 5)
    usage_trend_3m       = st.slider("Usage Trend 3M (%)", -50, 20, -10) / 100
    num_complaints       = st.selectbox("Number of Complaints", [0,1,2,3,4,5,6,7,8,9,10])
    complaint_resolution = st.selectbox("Last Complaint Status",
        ['No Complaints', 'Resolved', 'Pending', 'Escalated'])
    csat_score           = st.slider("CSAT Score (1-10)", 1.0, 10.0, 7.0, 0.5)
    nps_score            = st.slider("NPS Score (-100 to 100)", -100, 100, 20)
    num_support_calls    = st.selectbox("Support Calls (6 months)", list(range(0, 16)))
    avg_resolution_days  = st.slider("Avg Resolution Days", 0, 30, 3)

with col3:
    st.subheader("👤 Customer Profile & Billing")
    customer_age      = st.slider("Customer Age", 18, 70, 35)
    city_tier         = st.selectbox("City Tier", ['Metro', 'Tier2', 'Tier3'])
    income_segment    = st.selectbox("Income Segment", ['High', 'Mid', 'Low'])
    tech_savviness    = st.selectbox("Tech Savviness", ['High', 'Medium', 'Low'])
    referral_source   = st.selectbox("Referral Source",
        ['Organic', 'Referral', 'Advertisement', 'Promotion'])
    competitor_offer  = st.selectbox("Competitor Offer Available",
        [0, 1], format_func=lambda x: "Yes" if x else "No")
    payment_delays    = st.selectbox("Payment Delays", [0,1,2,3,4,5])
    discount_used     = st.selectbox("Currently on Discount",
        [0, 1], format_func=lambda x: "Yes" if x else "No")
    discount_expiry_days = st.slider("Discount Expires In (days)", 1, 180, 60) if discount_used else 999
    refund_requested  = st.selectbox("Refund Requested",
        [0, 1], format_func=lambda x: "Yes" if x else "No")
    engagement_index     = st.slider("Engagement Index (0-1)", 0.0, 1.0, 0.5, 0.05)
    feature_adoption_rate = st.slider("Feature Adoption Rate", 0.0, 1.0, 0.4, 0.05)

# ── Predict Button ─────────────────────────────────────────────
st.markdown("---")
pred_col, _ = st.columns([1, 3])
with pred_col:
    predict_btn = st.button("🔮 Predict Churn Risk", type="primary",
                             use_container_width=True)

# ── Prediction Results ─────────────────────────────────────────
if predict_btn:
    record = {
        'contract_type'       : contract_type,
        'plan_type'           : plan_type,
        'tenure_months'       : tenure_months,
        'payment_method'      : payment_method,
        'monthly_charges'     : monthly_charges,
        'total_charges'       : monthly_charges * tenure_months,
        'num_products'        : num_products,
        'paperless_billing'   : paperless_billing,
        'bill_increase_3m'    : bill_increase_3m,
        'days_to_renewal'     : days_to_renewal,
        'avg_data_usage_gb'   : avg_data_usage_gb,
        'last_30d_data_gb'    : last_30d_data_gb,
        'last_30d_calls'      : 80,
        'avg_monthly_calls'   : 100,
        'days_since_last_login': days_since_last_login,
        'usage_trend_3m'      : usage_trend_3m,
        'num_complaints'      : num_complaints,
        'complaint_resolution': complaint_resolution,
        'csat_score'          : csat_score,
        'nps_score'           : nps_score,
        'num_support_calls'   : num_support_calls,
        'avg_resolution_days' : avg_resolution_days,
        'customer_age'        : customer_age,
        'city_tier'           : city_tier,
        'income_segment'      : income_segment,
        'tech_savviness'      : tech_savviness,
        'referral_source'     : referral_source,
        'competitor_offer'    : competitor_offer,
        'payment_delays'      : payment_delays,
        'discount_used'       : discount_used,
        'discount_expiry_days': discount_expiry_days,
        'refund_requested'    : refund_requested,
        'engagement_index'    : engagement_index,
        'feature_adoption_rate': feature_adoption_rate,
        'roaming_usage'       : 0,
        'international_calls' : 0,
        'credit_card_expiry'  : 0,
        'num_devices'         : 2,
        'avg_session_duration': 25.0,
        'agent_quality_score' : 7.0,
        'last_complaint_days' : 30,
        'last_payment_days'   : 15,
        'disputed_bills'      : 0,
        'market_tenure_years' : 5.0,
        'usage_decline_score' : (last_30d_data_gb - avg_data_usage_gb) / (avg_data_usage_gb + 0.01),
        'complaint_intensity' : num_complaints / (tenure_months + 0.01),
        'value_score'         : monthly_charges * tenure_months / (tenure_months + 0.01),
    }

    with st.spinner("Analysing customer churn risk..."):
        try:
            result = predict(record)

            st.markdown("---")
            st.header("📊 Churn Prediction Results")

            # Main metrics
            m1, m2, m3, m4 = st.columns(4)
            churn_prob = result['churn_probability']

            with m1:
                st.metric("Churn Probability", f"{churn_prob:.1%}")
            with m2:
                risk_emoji = {'Critical':'🔴','High':'🟠','Medium':'🟡','Low':'🟢'}
                emoji = risk_emoji.get(result['risk_tier'], '⚪')
                st.metric("Risk Tier", f"{emoji} {result['risk_tier']}")
            with m3:
                st.metric("Prediction", result['churn_label'])
            with m4:
                st.metric("Model AUC", f"{result['model_info']['auc']}")

            # Risk box
            risk_class = f"risk-{result['risk_tier'].lower()}"
            st.markdown(f"""
            <div class="{risk_class}">
                <b>Retention Action:</b> {result['retention_action']}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")

            # Charts
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                # Probability gauge
                fig, ax = plt.subplots(figsize=(7, 4))
                colors   = ['#2e7d32', '#FFC107', '#FF5722', '#C00000']
                segments = [0.30, 0.20, 0.20, 0.30]
                labels   = ['Low\n<30%', 'Medium\n30-50%', 'High\n50-70%', 'Critical\n>70%']

                wedge_colors = []
                for i, (s, c) in enumerate(zip(segments, colors)):
                    wedge_colors.append(c if i == min(3, int(churn_prob/0.25)) else '#e0e0e0')

                ax.barh(['Risk'], [churn_prob], color=(
                    '#C00000' if churn_prob >= 0.70 else
                    '#FF5722' if churn_prob >= 0.50 else
                    '#FFC107' if churn_prob >= 0.30 else '#2e7d32'
                ), height=0.5, alpha=0.85)
                ax.barh(['Risk'], [1.0], color='#e8e8e8', height=0.5, alpha=0.3)
                ax.set_xlim(0, 1)
                ax.axvline(0.30, color='#FFC107',  linestyle='--', linewidth=1.5, alpha=0.7)
                ax.axvline(0.50, color='#FF5722', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.axvline(0.70, color='#C00000',  linestyle='--', linewidth=1.5, alpha=0.7)
                ax.text(churn_prob + 0.02, 0, f'{churn_prob:.1%}',
                        va='center', fontsize=14, fontweight='bold')
                ax.set_title('Churn Probability Gauge', fontweight='bold')
                ax.set_xlabel('Churn Probability')
                ax.set_yticks([])
                for x, lbl in zip([0.15, 0.40, 0.60, 0.85], labels):
                    ax.text(x, -0.4, lbl, ha='center', fontsize=8, color='#555')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with chart_col2:
                if result.get('shap_explanation'):
                    shap_data = result['shap_explanation']
                    features  = list(shap_data.keys())
                    values    = list(shap_data.values())

                    fig2, ax2 = plt.subplots(figsize=(7, 4))
                    colors2   = ['#C00000' if v > 0 else '#1F4E79' for v in values]
                    ax2.barh([f.replace('_',' ')[:20] for f in features][::-1],
                             values[::-1], color=colors2[::-1], alpha=0.85)
                    ax2.axvline(0, color='black', linewidth=1)
                    ax2.set_title('Why This Prediction?\n(SHAP Explanation)',
                                  fontweight='bold')
                    ax2.set_xlabel('Impact on Churn Probability')
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()

                    st.caption("🔴 Red = Increases churn risk | 🔵 Blue = Decreases churn risk")
                else:
                    st.info("SHAP explanation not available — install shap library")

            # ── LLM Retention Message ──────────────────────────────
            st.markdown("---")
            st.header("🤖 AI-Generated Retention Message")

            if openai_key:
                with st.spinner("Generating personalised retention message with OpenAI..."):
                    message = generate_retention_message(record, result, openai_key)

                if message and not message.startswith("LLM error") and not message.startswith("Install"):
                    st.markdown(f"""
                    <div class="llm-box">
                        <b>📱 Personalised Retention Message for this Customer:</b><br><br>
                        <i>"{message}"</i>
                    </div>
                    """, unsafe_allow_html=True)

                    # Show how it was generated
                    with st.expander("See how this message was generated"):
                        st.markdown(f"""
                        **Context used by AI:**
                        - Contract: {contract_type} | Tenure: {tenure_months} months
                        - Complaints: {num_complaints} | CSAT: {csat_score}
                        - Bill increase: {bill_increase_3m:.0%}
                        - Days to renewal: {days_to_renewal}
                        - Competitor offer: {'Yes' if competitor_offer else 'No'}
                        - Risk tier: {result['risk_tier']}
                        - Top SHAP drivers: {', '.join(list(result.get('shap_explanation', {}).keys())[:3])}
                        """)
                else:
                    st.error(f"Message generation failed: {message}")
            else:
                # Show example message without API key
                st.info("Add your OpenAI API key in the sidebar to generate personalised messages")
                st.markdown("""
                **Example message (what AI would generate):**

                > *"Hi there! We noticed your experience with us has been challenging lately —
                > we want to make it right. As a valued customer, we are applying a ₹150 credit
                > to your next bill and escalating your service complaint to our senior team
                > for resolution within 24 hours. You deserve better, and we are committed to delivering it."*
                """)

            # Model info
            st.markdown("---")
            st.caption(
                f"Model: {result['model_info']['model_name']} | "
                f"AUC: {result['model_info']['auc']} | "
                f"Threshold: {result['threshold_used']:.2f} | "
                f"Trained: {result['model_info']['trained_on']}"
            )

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Ensure best_model.pkl exists in S3: s3://customer-churn-project-2024/models/")

# ── Footer ──────────────────────────────────────────────────────
st.markdown("---")
st.caption("DV Analytics | Customer Churn MLOps | Author: Suresh D R")
