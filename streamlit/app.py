import json
import os
from pathlib import Path
from typing import Any, Dict

import requests
import streamlit as st

# -----------------------------------------------------------------------------
# MUST be the first Streamlit command
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Heart Disease Diagnosis", page_icon="❤️", layout="centered")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SCHEMA_PATH = Path("data/data_schema.json")

# API_URL is set in docker-compose environment or local default
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"

# -----------------------------------------------------------------------------
# Load schema from JSON file
# -----------------------------------------------------------------------------
@st.cache_resource
def load_schema(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


schema = load_schema(SCHEMA_PATH)

numerical_features = schema.get("numerical", {})
categorical_features = schema.get("categorical", {})

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("❤️ Heart Disease Prediction App")
st.write(
    f"This app sends clinical data to the FastAPI backend at **{API_BASE_URL}** for diagnostic inference."
)

st.header("Clinical Features")

user_input: Dict[str, Any] = {}

# -----------------------------------------------------------------------------
# Numerical Features
# -----------------------------------------------------------------------------
st.subheader("Numerical Features")

# Decide which heart features use sliders for better UX
SLIDER_FEATURES = {"age", "trestbps", "chol", "thalach", "oldpeak", "ca"}

for feature_name, stats in numerical_features.items():
    min_val = float(stats.get("min", 0.0))
    max_val = float(stats.get("max", 1000.0))
    mean_val = float(stats.get("mean", (min_val + max_val) / 2))
    median_val = float(stats.get("median", mean_val))

    # Use median as default
    default_val = median_val

    label = feature_name.replace("_", " ").title()
    help_text = (
        f"Min: {min_val:.2f}, Max: {max_val:.2f}, "
        f"Mean: {mean_val:.2f}, Median: {median_val:.2f}"
    )

    if feature_name in SLIDER_FEATURES:
        # Determine step size based on features
        if feature_name in {"age", "trestbps", "chol", "thalach", "ca"}:
            step = 1.0  # Discrete clinical values
        else:
            step = 0.1  # Fractional values like oldpeak

        user_input[feature_name] = st.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            value=float(default_val),
            step=step,
            help=help_text,
            key=feature_name,
        )
    else:
        # Fallback to number_input
        step = 1.0 if (max_val - min_val) > 10 else 0.1

        user_input[feature_name] = st.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=float(default_val),
            step=step,
            help=help_text,
            key=feature_name,
        )

# -----------------------------------------------------------------------------
# Categorical Features
# -----------------------------------------------------------------------------
st.subheader("Categorical Features")

for feature_name, info in categorical_features.items():
    unique_values = info.get("unique_values", [])
    value_counts = info.get("value_counts", {})

    if not unique_values:
        continue

    # Default to the most common value (Mode)
    if value_counts:
        default_value = max(value_counts, key=value_counts.get)
    else:
        default_value = unique_values[0]

    try:
        default_idx = unique_values.index(default_value)
    except ValueError:
        default_idx = 0

    label = feature_name.replace("_", " ").title()

    user_input[feature_name] = st.selectbox(
        label,
        options=unique_values,
        index=default_idx,
        key=feature_name,
        help=f"Training Distribution: {value_counts}",
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# Predict Button
# -----------------------------------------------------------------------------
if st.button("🔮 Run Diagnosis", type="primary"):
    # Note: We include a dummy patient_id as the API/Pipeline expects it
    user_input["patient_id"] = "WEB-UI-USER"
    payload = {"instances": [user_input]}

    with st.spinner("Analyzing clinical data..."):
        try:
            resp = requests.post(PREDICT_ENDPOINT, json=payload, timeout=30)
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request to API failed: {e}")
        else:
            if resp.status_code != 200:
                st.error(f"❌ API error: HTTP {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                preds = data.get("predictions", [])
                probs = data.get("probabilities", [])

                if not preds:
                    st.warning("⚠️ No predictions returned from API.")
                else:
                    pred = preds[0]
                    prob = probs[0] if probs else None
                    st.success("✅ Analysis Complete!")

                    st.subheader("Prediction Result")

                    # Logic to display Heart Disease Class and Probability
                    if pred == 1:
                        st.error(f"**Result: POSITIVE**")
                    else:
                        st.info(f"**Result: NEGATIVE**")

                    if prob is not None:
                        st.metric(label="Risk Confidence", value=f"{prob}%")
                    
                    # Show input summary in expander
                    with st.expander("📋 View Submitted Clinical Data"):
                        st.json(user_input)

st.markdown("---")
st.caption(
    f"📁 Schema: `{SCHEMA_PATH}`  \n"
    f"🌐 API: `{API_BASE_URL}`"
)