import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Purchase Prediction AI", page_icon="🛒", layout="wide")

MODEL_PATH = "model/purchase_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("🛒 Retail Customer Purchase Prediction")
st.caption("Predicts whether a website visitor is likely to complete a purchase, based on real-time browsing behavior.")

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run `python data/generate_data.py` then `python train_model.py` first.")
    st.stop()

model = load_model()

tab1, tab2 = st.tabs(["🔮 Live Prediction", "📊 Model Performance"])

with tab1:
    st.subheader("Enter customer session behavior")

    preset = st.selectbox(
        "Quick presets (optional)",
        ["Custom", "High-intent customer (Customer A)", "Low-intent customer (Customer B)"]
    )

    if preset == "High-intent customer (Customer A)":
        defaults = dict(pages=15, time=600, visits=4, products=8, cart="Yes", device="Desktop", prev_purchases=2)
    elif preset == "Low-intent customer (Customer B)":
        defaults = dict(pages=1, time=10, visits=0, products=0, cart="No", device="Mobile", prev_purchases=0)
    else:
        defaults = dict(pages=5, time=90, visits=1, products=3, cart="No", device="Mobile", prev_purchases=0)

    col1, col2 = st.columns(2)

    with col1:
        pages_viewed = st.slider("Pages viewed", 1, 30, defaults["pages"])
        time_spent_min = st.slider("Time spent on site (minutes)", 0.0, 20.0, defaults["time"] / 60, step=0.5)
        previous_visits = st.slider("Previous visits", 0, 15, defaults["visits"])
        products_viewed = st.slider("Products viewed", 0, 20, defaults["products"])

    with col2:
        added_to_cart = st.radio("Added product to cart?", ["Yes", "No"],
                                  index=0 if defaults["cart"] == "Yes" else 1, horizontal=True)
        device_type = st.selectbox("Device type", ["Mobile", "Desktop", "Tablet"],
                                    index=["Mobile", "Desktop", "Tablet"].index(defaults["device"]))
        previous_purchases = st.slider("Previous purchases", 0, 10, defaults["prev_purchases"])

    st.markdown("---")

    if st.button("🔍 Predict Purchase Likelihood", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            "pages_viewed": pages_viewed,
            "time_spent_sec": time_spent_min * 60,
            "previous_visits": previous_visits,
            "products_viewed": products_viewed,
            "added_to_cart": 1 if added_to_cart == "Yes" else 0,
            "device_type": device_type.lower(),
            "previous_purchases": previous_purchases,
        }])

        proba = model.predict_proba(input_df)[0][1]
        prediction = "Yes" if proba >= 0.5 else "No"

        colA, colB = st.columns([1, 2])
        with colA:
            if prediction == "Yes":
                st.success(f"### ✅ Purchase: {prediction}")
            else:
                st.error(f"### ❌ Purchase: {prediction}")
            st.metric("Purchase Probability", f"{proba*100:.1f}%")

        with colB:
            st.progress(min(max(proba, 0.0), 1.0))
            if proba >= 0.75:
                st.info("💡 **High-intent customer** — consider showing a limited-time discount or free-shipping nudge.")
            elif proba >= 0.4:
                st.info("💡 **Medium-intent customer** — a retargeting ad or exit-intent popup could help convert.")
            else:
                st.info("💡 **Low-intent customer** — likely still browsing; avoid aggressive upselling.")

with tab2:
    st.subheader("Model Comparison")
    if os.path.exists("model/model_comparison.csv"):
        comp_df = pd.read_csv("model/model_comparison.csv", index_col=0)
        st.dataframe(comp_df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("model/confusion_matrix.png"):
            st.image("model/confusion_matrix.png", caption="Confusion Matrix")
    with col2:
        if os.path.exists("model/roc_curve.png"):
            st.image("model/roc_curve.png", caption="ROC Curve")

    if os.path.exists("model/feature_importance.png"):
        st.image("model/feature_importance.png", caption="Feature Importance")

st.markdown("---")
st.caption("Built for hackathon demo · Model: scikit-learn · Framework: Streamlit")
