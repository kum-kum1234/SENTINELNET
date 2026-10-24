import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score

# =========================================================
# PAGE CONFIG + STYLE
# =========================================================
st.set_page_config(page_title="SentinelNet IDS Dashboard", layout="wide")

st.markdown("""
<style>
    .main { background:#020617; color:#e5e7eb; font-family:Inter,Segoe UI; }
    .big-header { background:linear-gradient(90deg,#0052D4,#4364F7,#6FB1FC); color:white;
                  padding:1.3rem 2rem; border-radius:18px; margin-bottom:1rem;
                  font-size:1.8rem; font-weight:800; }
    .metric-card { background:linear-gradient(135deg,#ffffff,#e3edff); border-radius:14px;
                   padding:1rem; text-align:center; color:#0f172a; font-weight:700; }
    .alerts-card { background:#ffe4e6; padding:0.6rem; border-radius:10px;
                   border-left:4px solid red; color:#7f1d1d; font-weight:600; margin-bottom:0.4rem; }
    .alerts-empty { font-size:1rem; opacity:0.7; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "monitoring_active" not in st.session_state:
    st.session_state.monitoring_active = True
if "cleared" not in st.session_state:
    st.session_state.cleared = False

# =========================================================
# MODEL DISCOVERY + GROUPING
# =========================================================
MODEL_DIR = Path("models")

@st.cache_resource
def discover_models():
    grouped = {
        ("NSL-KDD", "Binary"): {},
        ("NSL-KDD", "Multiclass"): {},
        ("CICIDS2017", "Binary"): {},
        ("CICIDS2017", "Multiclass"): {},
    }

    if not MODEL_DIR.exists():
        return grouped

    for fname in os.listdir(MODEL_DIR):
        if not fname.endswith(".pkl"):
            continue
        path = MODEL_DIR / fname

        f = fname.lower()

        if f.startswith("nslkdd_binary_"):
            key = ("NSL-KDD", "Binary")
            raw = fname[len("nslkdd_binary_"):-4]
        elif f.startswith("nslkdd_multi_") or f.startswith("nslkdd_multiclass_"):
            key = ("NSL-KDD", "Multiclass")
            raw = fname[len("nslkdd_multi_"):-4]
        elif f.startswith("cicids_binary_"):
            key = ("CICIDS2017", "Binary")
            raw = fname[len("cicids_binary_"):-4]
        elif f.startswith("cicids_multi_") or f.startswith("cicids_multiclass_"):
            key = ("CICIDS2017", "Multiclass")
            raw = fname[len("cicids_multi_"):-4]
        else:
            continue

        nice = raw.replace("_", " ").title()
        grouped[key][nice] = path

    return grouped

model_registry = discover_models()

# =========================================================
# SIDEBAR UI
# =========================================================
st.sidebar.markdown("<div class='sidebar-title'>⚙ Choose Setup</div>", unsafe_allow_html=True)

dataset = st.sidebar.selectbox("Dataset", ["NSL-KDD", "CICIDS2017"])
task = st.sidebar.radio("Task", ["Binary", "Multiclass"])
available_models = model_registry.get((dataset, task), {})

if not available_models:
    st.sidebar.error("🚨 No trained models found for your selection!")
    st.stop()

model_name = st.sidebar.selectbox("Select Model", list(available_models.keys()))
model_path = available_models[model_name]
model = joblib.load(model_path)

mode = st.sidebar.radio("Run Mode", ["File Analysis", "Live Monitoring"])

st.markdown("<div class='big-header'>SentinelNet Dashboard 🛡️💙</div>", unsafe_allow_html=True)

# =========================================================
# DATA INPUT + PREDICTION
# =========================================================
data_df, preds, y_true = None, None, None

if mode == "File Analysis":
    file = st.file_uploader("Upload your network-flow CSV", type=["csv"])
    if file: data_df = pd.read_csv(file)

elif mode == "Live Monitoring":
    file = st.file_uploader("Upload flow CSV to simulate live feed", type=["csv"], key="feed_csv")
    if file and st.session_state.monitoring_active:
        full_df = pd.read_csv(file)
        batch = st.slider("Latest flows batch size", 5, 300, 80)
        data_df = full_df.tail(batch)

# =========================================================
# RUN MODEL PREDICTION AFTER CSV LOAD
# =========================================================
if data_df is not None:
    raw = data_df.copy()

    # keep labels aside
    if "Label" in raw.columns:
        y_true = raw["Label"].values
        X = raw.drop(columns=["Label"])
    elif "labels" in raw.columns:
        y_true = raw["labels"].values
        X = raw.drop(columns=["labels"])
    else:
        X = raw.copy()

    # detect expected feature count dynamically
    expected = 78
    if hasattr(model, "n_features_in_"):
        expected = model.n_features_in_
    if hasattr(model, "get_booster"):
        expected = model.get_booster().num_features()

    # fix shape mismatch
    df_input = pd.DataFrame(X.copy())
    if df_input.shape[1] > expected:
        df_input = df_input.iloc[:, :expected]
    elif df_input.shape[1] < expected:
        pad = pd.DataFrame(np.zeros((len(df_input), expected - df_input.shape[1])))
        df_input = pd.concat([df_input.reset_index(drop=True), pad], axis=1)

    # optional scaling if model was trained scaled
    try:
        df_input = scaler.transform(df_input.values)
    except:
        df_input = df_input.values

    preds = model.predict(df_input)

    # Display predictions
    output = pd.DataFrame(df_input.copy())
    output["Prediction"] = preds
    st.write("### ✅ Model Predictions")
    st.dataframe(output)

    # Metrics
    total = len(preds)
    
    if task == "Binary":
        intr = int((preds == 1).sum())
    else:
        intr = int(sum([1 for p in preds if str(p).upper() != "NORMAL" and str(p).upper() != "BENIGN"]))

    norm = total - intr
    rate = (intr / total * 100) if total > 0 else 0

    # Accuracy print
    if y_true is not None:
        try:
            if task == "Binary":
                y_true_bin = np.array([0 if 'NORMAL' in str(l).upper() or 'BENIGN' in str(l).upper() else 1 for l in y_true])
                acc = accuracy_score(y_true_bin, preds)
                prec = precision_score(y_true_bin, preds, zero_division=0)
            else:
                acc = accuracy_score(y_true, preds)
                prec = None

            st.write(f"### 🎯 Test Accuracy: {acc*100:.2f}% ✅")
            if prec is not None:
                st.write(f"### 🎯 Test Precision: {prec*100:.2f}% ✅")

        except Exception:
            st.warning("⚠ Accuracy/precision calc failed due to format mismatch!")

    # Alert Section
    st.markdown("### ⚠ Recent Alerts")
    alerts = output[output["Prediction"] == 1] if task == "Binary" else output[output["Prediction"].astype(str).str.upper() != "BENIGN"]
    if alerts.empty:
        st.success("No intrusions detected ✅")
    else:
        for i, r in alerts.tail(8).iterrows():
            st.markdown(f"<div class='alerts-card'><span class='alerts-conn'>Flow #{i}</span> Attack detected ({r['Prediction']})</div>", unsafe_allow_html=True)
