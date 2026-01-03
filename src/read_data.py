import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "business_credit_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
print("Reading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

print("Columns in dataset:", df.columns.tolist())

# ---------------- ENCODING ----------------
business_encoder = LabelEncoder()
repayment_encoder = LabelEncoder()

df["business_type"] = business_encoder.fit_transform(df["business_type"])
df["repayment_history"] = repayment_encoder.fit_transform(df["repayment_history"])

# ---------------- TARGET ----------------
TARGET_COLUMN = "default_flag"   # ✅ CORRECT COLUMN

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------- SAVE FILES ----------------
joblib.dump(model, os.path.join(MODEL_DIR, "credit_risk_model.pkl"))
joblib.dump(business_encoder, os.path.join(MODEL_DIR, "business_encoder.pkl"))
joblib.dump(repayment_encoder, os.path.join(MODEL_DIR, "repayment_encoder.pkl"))

print("✅ MODEL AND ENCODERS SAVED SUCCESSFULLY")

