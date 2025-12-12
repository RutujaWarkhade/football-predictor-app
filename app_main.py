# app_main.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import tensorflow as tf
from pathlib import Path

# import predict_match 
try:
    from predict_match import predict_match
except Exception:
    predict_match = None


BASE = Path.cwd() 
MODELS_DIR = BASE / "models"

def find_first(existing_names):
    """Return Path of first existing filename from list (relative to BASE or models dir)."""
    for name in existing_names:
        p1 = BASE / name
        p2 = MODELS_DIR / name
        if p1.exists():
            return p1
        if p2.exists():
            return p2
    return None

def load_json_flexible(path_like):
    if path_like is None:
        return None
    try:
        with open(path_like, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

league_model_file = find_first(["rf_model.joblib", "rf_model.joblib", "rf_model.joblib", "rf_model.joblib", "rf_model.joblib"])
league_scaler_file = find_first(["scaler.joblib", "scaler.joblib", "feature_scaler.pkl", "feature_scaler.joblib"])
league_metadata_file = find_first(["model_metadata.json", "model_metadata", "model_metadata.json", "model_metadata"])
threshold_file = find_first(["threshold.json", "threshold", "threshold"])

# Goals & Assists pipelines and metadata (xgb_goals_pipeline.pkl and xgb_assists_pipeline.pkl)
goals_pipe = find_first(["xgb_goals_pipeline.pkl", "xgb_goals_pipeline.pkl", "xgb_goals_pipeline.pkl"])
assists_pipe = find_first(["xgb_assists_pipeline.pkl", "xgb_assists_pipeline.pkl", "xgb_assists_pipeline.pkl"])
meta_goals = find_first(["metadata_goals.json", "metadata_goals", "metadata_goals.json"])
meta_assists = find_first(["metadata_assists.json", "metadata_assists", "metadata_assists.json"])

# Match model files 
match_model_file = find_first(["models/best_football_predictor.h5", "best_football_predictor.h5", "models/best_football_predictor.h5"])
match_scaler_file = find_first(["models/feature_scaler.pkl", "feature_scaler.pkl", "models/feature_scaler.pkl"])

league_model = None
league_scaler = None
league_metadata = None
threshold_data = None
if league_model_file:
    try:
        league_model = joblib.load(league_model_file)
    except Exception:
        league_model = None

if league_scaler_file:
    try:
        league_scaler = joblib.load(league_scaler_file)
    except Exception:
        league_scaler = None

league_metadata = load_json_flexible(league_metadata_file)
threshold_data = load_json_flexible(threshold_file) or {}

# Goals / Assists pipelines
goals_model = None
assists_model = None
goals_meta_json = load_json_flexible(meta_goals)
assists_meta_json = load_json_flexible(meta_assists)
if goals_pipe:
    try:
        goals_model = joblib.load(goals_pipe)
    except Exception:
        goals_model = None
if assists_pipe:
    try:
        assists_model = joblib.load(assists_pipe)
    except Exception:
        assists_model = None

# Match model (TF .h5) and scaler
match_model = None
match_feature_scaler = None
if match_model_file:
    try:
        match_model = tf.keras.models.load_model(match_model_file)
    except Exception:
        match_model = None
if match_scaler_file:
    try:
        match_feature_scaler = joblib.load(match_scaler_file)
    except Exception:
        match_feature_scaler = None

st.set_page_config(page_title="Football Predictor Hub", layout="wide", page_icon="⚽")
st.markdown(
    """
    <style>
    .stApp { background-image: url("https://parimatch.co.tz/blog/wp-content/uploads/Football-predictions.jpg"); background-size: cover; background-position: center; background-attachment: fixed; }
    .overlay { background: rgba(0,0,0,0.55); border-radius: 14px; padding: 20px; color: white; backdrop-filter: blur(6px); }
    </style>
    """, unsafe_allow_html=True
)

#  Sidebar navigation 
st.sidebar.title("Football Predictor ⚽")
page = st.sidebar.radio("Navigate", ["Home", "Goals & Assists", "Match Winner", "League Winner"])

# PAGE: HOME 
if page == "Home":
    st.markdown(
        """
        <div class="overlay">
            <h1 style="text-align:center; font-size:50px;">⚽ Football Predictor Hub</h1>
            <p style="text-align:center; font-size:20px;">
                Analyze performance. Predict results. Lead the game! <br>
                Choose a prediction model from the sidebar 👈
            </p>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.markdown(
        """
        <div class="overlay" style="text-align:center;">
            <h3>🏆 League Winner</h3>
            <p>Predict which team will become champion.</p>
        </div>
        """, unsafe_allow_html=True
    )
    col2.markdown(
        """
        <div class="overlay" style="text-align:center;">
            <h3>🥅 Goals & Assists</h3>
            <p>Predict player performance statistics.</p>
        </div>
        """, unsafe_allow_html=True
    )
    col3.markdown(
        """
        <div class="overlay" style="text-align:center;">
            <h3>🎯 Match Winner</h3>
            <p>Predict match outcome & winning probabilities.</p>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="overlay" style="text-align:center; margin-top:25px;">
            <h4>🚀 Powered by Machine Learning & Real Football Data</h4>
        </div>
        """, unsafe_allow_html=True
    )


#  PAGE: GOALS & ASSISTS 
elif page == "Goals & Assists":
    st.markdown('<div class="overlay"><h2>Goals & Assists Predictor</h2></div>', unsafe_allow_html=True)

    if goals_model is None and assists_model is None:
        st.error("Goals/Assists model(s) not found. Expected pipeline files like xgb_goals_pipeline.pkl and xgb_assists_pipeline.pkl in project root.")
    else:
        
        target = st.radio("Choose target", ["Goals", "Assists"])
        meta = goals_meta_json if target == "Goals" else assists_meta_json
        model = goals_model if target == "Goals" else assists_model

        if meta is None or model is None:
            st.error(f"Missing metadata or model for {target}. Found model: {bool(model)}, metadata: {bool(meta)}")
        else:
        
            features = meta.get("features", [])
            st.markdown("Enter feature values (based on saved metadata)")
            input_vals = {}
            left, right = st.columns(2)
            for i, f in enumerate(features):
                col = left if (i % 2 == 0) else right
                ftype = meta.get("feature_types", {}).get(f, "numeric")
                if ftype == "categorical":
                    opts = meta.get("categories_map", {}).get(f, [])
                    if opts:
                        input_vals[f] = col.selectbox(f, opts)
                    else:
                        input_vals[f] = col.text_input(f, value="")
                else:
                    input_vals[f] = col.number_input(f, value=0.0)

            if st.button(f"Predict {target}"):
                X = pd.DataFrame([input_vals], columns=features)
                try:
                    pred = model.predict(X)[0]
                    st.success(f"Predicted {target}: {float(pred):.3f}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

#  PAGE: MATCH WINNER 
elif page == "Match Winner":
    st.markdown('<div class="overlay"><h2>Match Winner Predictor</h2></div>', unsafe_allow_html=True)

    # use the features used in your train script
    match_features = [
        'HTWinStreak3','HTWinStreak5','HTLossStreak3','HTLossStreak5',
        'ATWinStreak3','ATWinStreak5','ATLossStreak3','ATLossStreak5',
        'HTP','ATP','HTFormPts','ATFormPts','HTGD','ATGD','DiffPts','DiffFormPts'
    ]

    st.markdown("Enter match-level features:")
    match_input = {}
    cols = st.columns(2)
    for i, f in enumerate(match_features):
        col = cols[i % 2]
        match_input[f] = col.number_input(f, value=0.0)

    if st.button("Predict Match Outcome"):
        
        if predict_match is not None:
            try:
                result, prob = predict_match([match_input[f] for f in match_features])
                st.success(f"Prediction: {'Home Win' if result in (1, '1', 'Home') else result}")
                st.info(f"Home Win Probability: {prob:.3f}")
            except Exception as e:
                st.error(f"predict_match imported but failed: {e}\nFalling back to TF model if present.")
        else:
            
            if match_model is None or match_feature_scaler is None:
                st.error("No match prediction function and no TF model+scaler found. Expected files: models/best_football_predictor.h5 and models/feature_scaler.pkl")
            else:
                try:
                    X = pd.DataFrame([match_input])[match_features]
                    Xs = match_feature_scaler.transform(X)
                    proba = float(match_model.predict(Xs)[0][0])
                    st.success(f"Home Win Probability: {proba:.3f}")
                    st.info("Home Win" if proba > 0.5 else "Not Home Win")
                except Exception as e:
                    st.error(f"TF model prediction failed: {e}")

#  PAGE: LEAGUE WINNER 
elif page == "League Winner":
    st.markdown('<div class="overlay"><h2>League Winner Predictor</h2></div>', unsafe_allow_html=True)

    if league_model is None or league_scaler is None or league_metadata is None:
        st.error("League model/scaler/metadata not found. Expected rf_model.joblib, scaler.joblib (or feature_scaler.pkl) and model_metadata.json at project root.")
        st.write("Detected files:")
        st.write({
            "league_model": str(league_model_file) if league_model_file else None,
            "league_scaler": str(league_scaler_file) if league_scaler_file else None,
            "league_metadata": str(league_metadata_file) if league_metadata_file else None,
            "threshold": str(threshold_file) if threshold_file else None
        })
    else:
        feats = league_metadata.get("feature_columns", [])
        inputs = {}
        left, right = st.columns(2)
        for i, f in enumerate(feats):
            col = left if (i % 2 == 0) else right
            inputs[f] = col.number_input(f, value=0.0)

        if st.button("Predict Champion Probability"):
            try:
                X = pd.DataFrame([inputs])[feats]
                Xs = league_scaler.transform(X)
                proba = float(league_model.predict_proba(Xs)[:, 1][0])
                
                opt_thresh = threshold_data.get("optimal_threshold", 0.5)
                st.metric("Champion Probability", f"{proba:.3f}")
                st.success("Champion" if proba > opt_thresh else "Not Champion")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("App loaded from: " + str(BASE))