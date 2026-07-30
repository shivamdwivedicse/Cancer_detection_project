import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Breast Cancer Detection", page_icon="🩺", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("cancer_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
    "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
]

# Reasonable default values (dataset means) so the form isn't empty
DEFAULTS = {
    "radius_mean": 14.13, "texture_mean": 19.29, "perimeter_mean": 91.97,
    "area_mean": 654.89, "smoothness_mean": 0.096, "compactness_mean": 0.104,
    "concavity_mean": 0.089, "concave points_mean": 0.048, "symmetry_mean": 0.181,
    "fractal_dimension_mean": 0.063, "radius_se": 0.405, "texture_se": 1.217,
    "perimeter_se": 2.866, "area_se": 40.34, "smoothness_se": 0.007,
    "compactness_se": 0.025, "concavity_se": 0.032, "concave points_se": 0.012,
    "symmetry_se": 0.021, "fractal_dimension_se": 0.004, "radius_worst": 16.27,
    "texture_worst": 25.68, "perimeter_worst": 107.26, "area_worst": 880.58,
    "smoothness_worst": 0.132, "compactness_worst": 0.254, "concavity_worst": 0.272,
    "concave points_worst": 0.114, "symmetry_worst": 0.290, "fractal_dimension_worst": 0.084,
}

GROUPS = {
    "Mean values": [f for f in FEATURES if f.endswith("_mean")],
    "Standard Error values": [f for f in FEATURES if f.endswith("_se")],
    "Worst values": [f for f in FEATURES if f.endswith("_worst")],
}

st.title("🩺 Breast Cancer Prediction (Malignant vs Benign)")
st.caption(
    "Enter the tumor's cell nuclei measurements below. The model predicts "
    "whether the mass is likely **Malignant** or **Benign**."
)
st.info(
    "⚠️ This tool is for educational/demo purposes only and is **not** a medical "
    "diagnostic device. Always consult a qualified doctor.",
    icon="⚠️",
)

with st.form("prediction_form"):
    inputs = {}
    tabs = st.tabs(list(GROUPS.keys()))
    for tab, (group_name, feats) in zip(tabs, GROUPS.items()):
        with tab:
            cols = st.columns(2)
            for i, feat in enumerate(feats):
                with cols[i % 2]:
                    inputs[feat] = st.number_input(
                        feat.replace("_", " ").title(),
                        min_value=0.0,
                        value=float(DEFAULTS[feat]),
                        format="%.5f",
                        key=feat,
                    )
    submitted = st.form_submit_button("🔍 Predict", use_container_width=True, type="primary")

if submitted:
    x = np.array([[inputs[f] for f in FEATURES]])
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    proba = model.predict_proba(x_scaled)[0]

    st.divider()
    if pred == 1:
        st.error(f"### Result: Malignant\nConfidence: **{proba[1]*100:.2f}%**")
    else:
        st.success(f"### Result: Benign\nConfidence: **{proba[0]*100:.2f}%**")

    st.write("Prediction probabilities:")
    st.progress(float(proba[1]), text=f"Malignant probability: {proba[1]*100:.2f}%")
    st.progress(float(proba[0]), text=f"Benign probability: {proba[0]*100:.2f}%")

st.divider()
st.caption("Model: Logistic Regression (C=0.5, class_weight='balanced') · ~94.9% CV accuracy")