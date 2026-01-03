from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

# ---------------- APP INITIALIZATION ----------------
app = Flask(__name__)

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "credit_risk_model.pkl")
BUSINESS_ENCODER_PATH = os.path.join(MODEL_DIR, "business_encoder.pkl")
REPAYMENT_ENCODER_PATH = os.path.join(MODEL_DIR, "repayment_encoder.pkl")

# ---------------- LOAD MODEL & ENCODERS ----------------
model = joblib.load(MODEL_PATH)
business_encoder = joblib.load(BUSINESS_ENCODER_PATH)
repayment_encoder = joblib.load(REPAYMENT_ENCODER_PATH)

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET"])
def home():
    return "AI Credit Evaluation API is running"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # ---------------- INPUT VALIDATION ----------------
    if not (300 <= data["credit_score"] <= 900):
        return jsonify({"error": "Credit score must be between 300 and 900"}), 400

    if not (0 <= data["debt_to_income_ratio"] <= 1):
        return jsonify({"error": "Debt-to-income ratio must be between 0 and 1"}), 400

    if data["loan_amount_requested"] <= 0:
        return jsonify({"error": "Loan amount must be greater than 0"}), 400

    # ---------------- INPUT PREPARATION ----------------
    input_df = pd.DataFrame([{
        "business_type": business_encoder.transform([data["business_type"]])[0],
        "years_in_operation": int(data["years_in_operation"]),
        "annual_revenue": float(data["annual_revenue"]),
        "monthly_cashflow": float(data["monthly_cashflow"]),
        "loan_amount_requested": float(data["loan_amount_requested"]),
        "credit_score": int(data["credit_score"]),
        "existing_loans": int(data["existing_loans"]),
        "debt_to_income_ratio": float(data["debt_to_income_ratio"]),
        "collateral_value": float(data["collateral_value"]),
        "repayment_history": repayment_encoder.transform(
            [data["repayment_history"]]
        )[0]
    }])

    # ---------------- MODEL PREDICTION ----------------
    probability = model.predict_proba(input_df)[0][1]
    risk_score = round(probability * 100, 2)

    # ---------------- DECISION LOGIC ----------------
    if risk_score < 30:
        decision = "APPROVE"
        risk_category = "Low Risk"
    elif risk_score < 60:
        decision = "REVIEW"
        risk_category = "Moderate Risk"
    else:
        decision = "REJECT"
        risk_category = "High Risk"

    # ---------------- CONFIDENCE LEVEL ----------------
    if probability < 0.2 or probability > 0.8:
        confidence = "High"
    elif probability < 0.4 or probability > 0.6:
        confidence = "Medium"
    else:
        confidence = "Low"

    # ---------------- DECISION EXPLANATION ----------------
    reasons = []

    if data["credit_score"] < 650:
        reasons.append("Low credit score")

    if data["debt_to_income_ratio"] > 0.5:
        reasons.append("High debt-to-income ratio")

    if data["loan_amount_requested"] > data["annual_revenue"]:
        reasons.append("Loan amount exceeds annual revenue")

    if data["repayment_history"].lower() == "poor":
        reasons.append("Poor repayment history")

    if not reasons:
        reasons.append("Strong financial profile")

    # ---------------- RESPONSE ----------------
    return jsonify({
        "risk_score": risk_score,
        "risk_category": risk_category,
        "decision": decision,
        "confidence": confidence,
        "reasons": reasons
    })


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

