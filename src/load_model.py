import joblib
import pandas as pd

# Load saved model and encoders
model = joblib.load("model/credit_risk_model.pkl")
le_business = joblib.load("model/business_encoder.pkl")
le_repayment = joblib.load("model/repayment_encoder.pkl")

print("Model and encoders loaded successfully.")

# New applicant (same format as before)
new_applicant = pd.DataFrame([{
    "business_type": le_business.transform(["Retail"])[0],
    "years_in_operation": 4,
    "annual_revenue": 800000,
    "monthly_cashflow": 60000,
    "loan_amount_requested": 250000,
    "credit_score": 700,
    "existing_loans": 1,
    "debt_to_income_ratio": 0.35,
    "collateral_value": 350000,
    "repayment_history": le_repayment.transform(["Average"])[0]
}])

# Predict
prediction = model.predict(new_applicant)
probability = model.predict_proba(new_applicant)

print("\nLoaded Model Prediction:")
print("Default Flag:", prediction[0])
print("Probability [No Default, Default]:", probability[0])
