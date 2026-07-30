import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("Cancer_Data.csv")
df = df.drop(columns=["id", "Unnamed: 32"])
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

y = df["diagnosis"]

# ---------------------------------------------------------------
# 1. FULL MODEL (30 features) - for "Advanced Mode"
# ---------------------------------------------------------------
x_full = df.drop(columns=["diagnosis"])
xt, xte, yt, yte = train_test_split(x_full, y, test_size=0.20, random_state=42)

scaler_full = StandardScaler()
xt_s = scaler_full.fit_transform(xt)
xte_s = scaler_full.transform(xte)

model_full = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")
model_full.fit(xt_s, yt)

print("=== FULL MODEL (30 features) ===")
print("Test Accuracy:", accuracy_score(yte, model_full.predict(xte_s)))
print(confusion_matrix(yte, model_full.predict(xte_s)))
cv_full = cross_val_score(model_full, scaler_full.fit_transform(x_full), y, cv=5)
print("CV mean:", cv_full.mean())

joblib.dump(model_full, "cancer_model.pkl")
joblib.dump(scaler_full, "scaler.pkl")

# ---------------------------------------------------------------
# 2. SIMPLIFIED MODEL (top 10 features) - for "Simple Mode"
#    Features picked by |coefficient| importance from the full model
# ---------------------------------------------------------------
TOP10_FEATURES = [
    "texture_worst", "radius_se", "symmetry_worst", "concave points_mean",
    "radius_worst", "concavity_worst", "concave points_worst", "area_worst",
    "area_se", "concavity_mean",
]

x_simple = df[TOP10_FEATURES]
xt2, xte2, yt2, yte2 = train_test_split(x_simple, y, test_size=0.20, random_state=42)

scaler_simple = StandardScaler()
xt2_s = scaler_simple.fit_transform(xt2)
xte2_s = scaler_simple.transform(xte2)

model_simple = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")
model_simple.fit(xt2_s, yt2)

print("\n=== SIMPLIFIED MODEL (10 features) ===")
print("Test Accuracy:", accuracy_score(yte2, model_simple.predict(xte2_s)))
print(confusion_matrix(yte2, model_simple.predict(xte2_s)))
cv_simple = cross_val_score(model_simple, scaler_simple.fit_transform(x_simple), y, cv=5)
print("CV mean:", cv_simple.mean())

joblib.dump(model_simple, "cancer_model_simple.pkl")
joblib.dump(scaler_simple, "scaler_simple.pkl")

print("\nSaved: cancer_model.pkl, scaler.pkl, cancer_model_simple.pkl, scaler_simple.pkl")
