
# AI Credit Evaluation System (Case 1)

## Project Overview
The AI Credit Evaluation System is a machine learning–based application designed to assess the credit risk of business loan applicants.  
The system predicts whether a business is likely to default on a loan and provides a decision recommendation: **APPROVE, REVIEW, or REJECT**.

This project is implemented as part of **Case 1 – AI‑Driven Smart Credit Evaluation System** and demonstrates a complete end‑to‑end machine learning workflow, from data preprocessing to model deployment using a Flask API.

---

## Objectives
- Automate the credit risk assessment process
- Predict the probability of loan default for businesses
- Assist financial institutions in loan decision‑making
- Deploy a machine learning model using a REST API

---

## Machine Learning Approach
- **Problem Type:** Binary Classification  
- **Algorithm Used:** Logistic Regression  
- **Target Variable:** `default_flag`  
  - `0` → No default (low risk)  
  - `1` → Default (high risk)

The model outputs a probability of default, which is converted into a **risk score (0–100)** and mapped to a decision.

---

## Dataset Description
The dataset used is `business_credit_data.csv`, which contains historical business credit information.

### Features
- business_type  
- years_in_operation  
- annual_revenue  
- monthly_cashflow  
- loan_amount_requested  
- credit_score  
- existing_loans  
- debt_to_income_ratio  
- collateral_value  
- repayment_history  

### Unique Feature – Decision Explanation
The system provides human‑readable explanations along with each credit decision, improving transparency and interpretability of the ML model.


### Target
- default_flag

---

## Project Structure
=======
# AI-credit-evaluation
AI-based Credit Risk Evaluation System 

