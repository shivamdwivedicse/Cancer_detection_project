import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Breast Cancer Detection", page_icon="🩺", layout="wide")

@st.cache_resource
def load_artifacts():
    model_full = joblib.load("cancer_model.pkl")
    scaler_full = joblib.load("scaler.pkl")
    model_simple = joblib.load("cancer_model_simple.pkl")
    scaler_simple = joblib.load("scaler_simple.pkl")
    return model_full, scaler_full, model_simple, scaler_simple

model_full, scaler_full, model_simple, scaler_simple = load_artifacts()

FEATURES_FULL = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
    "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
]

# The 10 most influential features (picked by |coefficient| from the full model)
FEATURES_SIMPLE = [
    "texture_worst", "radius_se", "symmetry_worst", "concave points_mean",
    "radius_worst", "concavity_worst", "concave points_worst", "area_worst",
    "area_se", "concavity_mean",
]

# Dataset-mean default values, so the form starts pre-filled with something reasonable
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

# Human-friendly label + plain-language help text for every feature
LABELS = {
    "radius_mean": ("Radius (Average)", "Average distance from the tumor's center to its edge."),
    "texture_mean": ("Texture (Average)", "How rough/varied the cell surface looks (grayscale variation)."),
    "perimeter_mean": ("Perimeter (Average)", "Average length around the tumor's boundary."),
    "area_mean": ("Area (Average)", "Average area covered by the tumor."),
    "smoothness_mean": ("Smoothness (Average)", "How smooth vs. jagged the tumor's edge is."),
    "compactness_mean": ("Compactness (Average)", "How tightly packed / dense the tumor's shape is."),
    "concavity_mean": ("Concavity (Average)", "How deep the indentations (dents) in the tumor's edge are."),
    "concave points_mean": ("Concave Points (Average)", "Number of dents/indented points on the tumor's edge."),
    "symmetry_mean": ("Symmetry (Average)", "How symmetrical the tumor's shape is."),
    "fractal_dimension_mean": ("Fractal Dimension (Average)", "How complex/irregular the boundary pattern is."),
    "radius_se": ("Radius (Variation)", "How much the radius varies across measurements."),
    "texture_se": ("Texture (Variation)", "How much the texture varies across measurements."),
    "perimeter_se": ("Perimeter (Variation)", "How much the perimeter varies across measurements."),
    "area_se": ("Area (Variation)", "How much the tumor's area varies across measurements."),
    "smoothness_se": ("Smoothness (Variation)", "How much the smoothness varies across measurements."),
    "compactness_se": ("Compactness (Variation)", "How much the compactness varies across measurements."),
    "concavity_se": ("Concavity (Variation)", "How much the concavity varies across measurements."),
    "concave points_se": ("Concave Points (Variation)", "How much the number of dents varies."),
    "symmetry_se": ("Symmetry (Variation)", "How much the symmetry varies across measurements."),
    "fractal_dimension_se": ("Fractal Dimension (Variation)", "How much the boundary complexity varies."),
    "radius_worst": ("Radius (Worst / Largest)", "The largest radius value observed in the sample."),
    "texture_worst": ("Texture (Worst / Roughest)", "The most extreme (roughest) texture value observed."),
    "perimeter_worst": ("Perimeter (Worst / Largest)", "The largest perimeter value observed."),
    "area_worst": ("Area (Worst / Largest)", "The largest area value observed."),
    "smoothness_worst": ("Smoothness (Worst)", "The most extreme smoothness value observed."),
    "compactness_worst": ("Compactness (Worst)", "The most extreme compactness value observed."),
    "concavity_worst": ("Concavity (Worst / Deepest)", "The deepest indentation value observed."),
    "concave points_worst": ("Concave Points (Worst / Most)", "The highest number of dents observed."),
    "symmetry_worst": ("Symmetry (Worst / Least Symmetric)", "The least symmetrical value observed."),
    "fractal_dimension_worst": ("Fractal Dimension (Worst)", "The most complex boundary value observed."),
}

GROUPS_FULL = {
    "Average values": [f for f in FEATURES_FULL if f.endswith("_mean")],
    "Variation values": [f for f in FEATURES_FULL if f.endswith("_se")],
    "Worst-case values": [f for f in FEATURES_FULL if f.endswith("_worst")],
}

st.title("🩺 Breast Cancer Prediction (Malignant vs Benign)")
st.caption(
    "Enter the tumor's cell nuclei measurements below (usually taken from a lab report / "
    "digitized image analysis). The model predicts whether the mass is likely "
    "**Malignant** or **Benign**."
)
st.info(
    "⚠️ This tool is for educational/demo purposes only and is **not** a medical "
    "diagnostic device. Always consult a qualified doctor.",
    icon="⚠️",
)

mode = st.radio(
    "Choose a mode:",
    ["🟢 Simple (10 key values, recommended)", "🔵 Advanced (all 30 values)"],
    horizontal=True,
)
is_simple = mode.startswith("🟢")

if is_simple:
    st.caption(
        "Simple mode uses the **10 most important measurements** (picked automatically "
        "by the model). Accuracy: ~96.5% — close to the full model, but much quicker to fill."
    )
else:
    st.caption("Advanced mode uses all 30 measurements. Accuracy: ~97.5%.")

with st.form("prediction_form"):
    inputs = {}

    if is_simple:
        cols = st.columns(2)
        for i, feat in enumerate(FEATURES_SIMPLE):
            label, help_text = LABELS[feat]
            with cols[i % 2]:
                inputs[feat] = st.number_input(
                    label,
                    min_value=0.0,
                    value=float(DEFAULTS[feat]),
                    format="%.5f",
                    help=help_text,
                    key="simple_" + feat,
                )
    else:
        tabs = st.tabs(list(GROUPS_FULL.keys()))
        for tab, (group_name, feats) in zip(tabs, GROUPS_FULL.items()):
            with tab:
                cols = st.columns(2)
                for i, feat in enumerate(feats):
                    label, help_text = LABELS[feat]
                    with cols[i % 2]:
                        inputs[feat] = st.number_input(
                            label,
                            min_value=0.0,
                            value=float(DEFAULTS[feat]),
                            format="%.5f",
                            help=help_text,
                            key="full_" + feat,
                        )

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True, type="primary")

if submitted:
    if is_simple:
        x = np.array([[inputs[f] for f in FEATURES_SIMPLE]])
        x_scaled = scaler_simple.transform(x)
        pred = model_simple.predict(x_scaled)[0]
        proba = model_simple.predict_proba(x_scaled)[0]
    else:
        x = np.array([[inputs[f] for f in FEATURES_FULL]])
        x_scaled = scaler_full.transform(x)
        pred = model_full.predict(x_scaled)[0]
        proba = model_full.predict_proba(x_scaled)[0]

    st.divider()
    if pred == 1:
        st.error(f"### Result: Malignant\nConfidence: **{proba[1]*100:.2f}%**")
    else:
        st.success(f"### Result: Benign\nConfidence: **{proba[0]*100:.2f}%**")

    st.write("Prediction probabilities:")
    st.progress(float(proba[1]), text=f"Malignant probability: {proba[1]*100:.2f}%")
    st.progress(float(proba[0]), text=f"Benign probability: {proba[0]*100:.2f}%")

st.divider()
st.caption("Model: Logistic Regression (C=0.5, class_weight='balanced')")
