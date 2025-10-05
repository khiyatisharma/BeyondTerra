# Save as app.py and run: streamlit run app.py
import streamlit as st
import numpy as np
import joblib
import shap
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Beyond Terra: Exoplanet Predictor", layout="centered")

st.title("🪐 Beyond Terra: Exoplanet Predictor")
st.markdown("""
Welcome to **Beyond Terra**, an AI-powered tool built for the NASA Space Apps Challenge 2025.  
This app uses a trained Random Forest model on **Kepler exoplanet data** to predict whether a given object is likely a planet or not.
""")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    clf = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return clf, scaler

clf, scaler = load_model()

# -------------------- DEFINE FEATURES --------------------
features = [
    'koi_steff', 'koi_slogg', 'koi_srad', 'koi_period', 'koi_prad',
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
]

# -------------------- INPUT SECTION --------------------
st.subheader("🔭 Input Features")
st.caption("Enter the observed parameters below:")

col1, col2, col3 = st.columns(3)
inputs = []

for i, feat in enumerate(features):
    with [col1, col2, col3][i % 3]:
        val = st.number_input(f"{feat}", value=0.0, format="%.5f")
        inputs.append(val)

# -------------------- PREDICTION --------------------
if st.button("🚀 Predict Exoplanet"):
    input_scaled = scaler.transform([inputs])
    pred = clf.predict(input_scaled)[0]
    prob = clf.predict_proba(input_scaled)[0][pred]

    st.markdown("---")
    if pred == 1:
        st.success(f"✅ **Result:** This is likely a *confirmed planet*! (Confidence: {prob*100:.2f}%)")
    else:
        st.error(f"❌ **Result:** This is likely *not a planet*. (Confidence: {prob*100:.2f}%)")

    st.markdown("""
    **How it works:**  
    Our model uses key stellar and orbital parameters (like surface temperature, radius, and orbital period)
    to classify potential exoplanets using a **Random Forest machine learning algorithm** trained on NASA Kepler data.
    """)

st.markdown("---")
st.caption("Created by Team **Beyond Terra** | NASA Space Apps Challenge 2025 🌍")

# -------------------- LOAD & PREPARE DATA --------------------
@st.cache_data
def load_data():
    data = pd.read_csv("cumulative_2025.10.04_21.52.29.csv", comment='#')
    data['label'] = data['koi_disposition'].map({
        'CONFIRMED': 1, 'CANDIDATE': 0, 'FALSE POSITIVE': 0
    })
    X = data[features].fillna(data[features].median())
    y = data['label']
    X_scaled = scaler.transform(X)
    return X, X_scaled, y

X, X_scaled, y = load_data()

# -------------------- MODEL EVALUATION --------------------
y_pred = clf.predict(X_scaled)
acc = accuracy_score(y, y_pred)
report = classification_report(y, y_pred, output_dict=True)
cm = confusion_matrix(y, y_pred)
feat_importances = pd.Series(clf.feature_importances_, index=features)

# -------------------- EXPANDERS --------------------
with st.expander("📊 Model Performance"):
    st.metric("Accuracy", f"{acc*100:.2f}%")
    st.write("Classification Report:")
    st.json(report)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Planet', 'Planet'],
                yticklabels=['Not Planet', 'Planet'], ax=ax)
    plt.title("Confusion Matrix")
    st.pyplot(fig)

with st.expander("📈 Feature Importance"):
    fig, ax = plt.subplots()
    feat_importances.sort_values().plot(kind='barh', ax=ax)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")
    st.pyplot(fig)

with st.expander("🔍 SHAP Feature Impact"):
    st.caption("This explains which features most influence each prediction.")

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_scaled)

    if isinstance(shap_values, list):
        shap_to_plot = shap_values[1]
    else:
        shap_to_plot = shap_values 

    X_df = pd.DataFrame(X_scaled, columns=features)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_to_plot, X_df, show=False)
    st.pyplot(fig, bbox_inches="tight")

