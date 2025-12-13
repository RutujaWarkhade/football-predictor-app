# app_main.py - COMPLETE UPDATED VERSION FOR STREAMLIT CLOUD
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from pathlib import Path

# ============================================
# SKLEARN COMPATIBILITY FIX
# ============================================
try:
    import sklearn
    # Check if _RemainderColsList exists, if not create it
    if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
        class _RemainderColsList(list):
            pass
        sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
except:
    pass

# Try to import tensorflow, but don't crash if it fails
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    st.warning("TensorFlow not available - Match predictor will be disabled")

# ============================================
# STREAMLIT CLOUD FILE HANDLING
# ============================================

# Get base directory - Streamlit Cloud runs from /home/appuser/venv/lib/python3.9/site-packages
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Debug info (commented out for production)
# st.write(f"Base directory: {BASE_DIR}")
# st.write(f"Files in directory: {os.listdir(BASE_DIR)}")

def find_file_simple(filename):
    """Simple file finder for Streamlit Cloud"""
    locations = [
        filename,  # Current directory
        BASE_DIR / filename,  # Base directory
        BASE_DIR / "models" / filename,  # Models folder
        BASE_DIR / "metadata" / filename,  # Metadata folder
        Path("models") / filename,  # Relative models folder
        Path("metadata") / filename,  # Relative metadata folder
    ]
    
    for location in locations:
        if os.path.exists(location):
            return location
    return None

def load_model_safe(filename):
    """Safe model loader with error handling"""
    try:
        file_path = find_file_simple(filename)
        if file_path is None:
            st.warning(f"File not found: {filename}")
            return None
        
        # Check file extension
        if filename.endswith('.h5'):
            if TENSORFLOW_AVAILABLE:
                return tf.keras.models.load_model(file_path)
            else:
                st.warning(f"Cannot load {filename}: TensorFlow not available")
                return None
        elif filename.endswith(('.pkl', '.joblib')):
            # Try different loading methods
            try:
                return joblib.load(file_path)
            except Exception as e1:
                try:
                    import pickle
                    with open(file_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e2:
                    try:
                        import cloudpickle
                        with open(file_path, 'rb') as f:
                            return cloudpickle.load(f)
                    except Exception as e3:
                        st.warning(f"Failed to load {filename} with all methods")
                        return None
        else:
            return joblib.load(file_path)
    except Exception as e:
        st.warning(f"Error loading {filename}: {str(e)[:100]}...")
        return None

def load_json_safe(filename):
    """Safe JSON loader"""
    try:
        file_path = find_file_simple(filename)
        if file_path is None:
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Error loading JSON {filename}: {e}")
        return None

# ============================================
# LOAD MODELS WITH PROGRESS AND ERROR HANDLING
# ============================================

# Initialize all models as None
league_model = None
league_scaler = None
goals_model = None
assists_model = None
match_model = None
match_feature_scaler = None
league_metadata = None
threshold_data = {}
goals_meta_json = None
assists_meta_json = None

# Try to load models, but don't crash if any fail
try:
    # League Winner Model
    league_model = load_model_safe("rf_model.joblib")
    league_scaler = load_model_safe("scaler.joblib")
    league_metadata = load_json_safe("model_metadata.json")
    threshold_data = load_json_safe("threshold.json") or {}
    
    # Goals & Assists Models
    goals_model = load_model_safe("xgb_goals_pipeline.pkl")
    assists_model = load_model_safe("xgb_assists_pipeline.pkl")
    goals_meta_json = load_json_safe("metadata_goals.json")
    assists_meta_json = load_json_safe("metadata_assists.json")
    
    # Match Winner Model
    if TENSORFLOW_AVAILABLE:
        match_model = load_model_safe("best_football_predictor.h5")
        match_feature_scaler = load_model_safe("feature_scaler.pkl")
except Exception as e:
    st.warning(f"Some models failed to load: {str(e)[:100]}...")

# ============================================
# STREAMLIT APP CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Football Predictor Pro", 
    layout="wide", 
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

# Custom CSS with beautiful green and black theme
st.markdown(
    """
    <style>
    /* Main background - Green/Black theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #0d1f0d 50%, #0a0a0a 100%);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Overlay cards - Dark with green accents */
    .overlay-card {
        background: rgba(10, 20, 10, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        color: #ffffff;
        border: 1px solid rgba(46, 204, 113, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .overlay-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(46, 204, 113, 0.2);
        border: 1px solid rgba(46, 204, 113, 0.4);
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Buttons - Green gradient */
    .stButton > button {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        color: white;
        border: none;
        padding: 14px 28px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(46, 204, 113, 0.4);
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(10, 20, 10, 0.9);
        border-right: 1px solid rgba(46, 204, 113, 0.1);
    }
    
    /* Success messages */
    .stSuccess {
        background: rgba(46, 204, 113, 0.15);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #2ecc71;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(46, 204, 113, 0); }
        100% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0); }
    }
    
    /* Metrics and cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.1), rgba(46, 204, 113, 0.05));
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 15px 0;
        border: 1px solid rgba(46, 204, 113, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
        border: 1px solid rgba(46, 204, 113, 0.4);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(46, 204, 113, 0.3);
        color: white;
        border-radius: 8px;
    }
    
    /* Headers - Green gradient text */
    h1, h2, h3 {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Emojis with original colors */
    .emoji {
        font-family: "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif;
        color: inherit !important;
    }
    
    /* Card emojis */
    .card-emoji {
        font-size: 3rem;
        margin-bottom: 15px;
        display: block;
    }
    
    /* Dashboard emojis */
    .dashboard-emoji {
        font-size: 3rem;
        margin-bottom: 15px;
        display: block;
    }
    
    /* Sidebar emojis */
    .sidebar-emoji {
        font-size: 2.5rem;
        margin-bottom: 15px;
        display: block;
    }
    
    /* Football field animation */
    .football-field {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
        opacity: 0.05;
    }
    
    .field-line {
        position: absolute;
        background: rgba(46, 204, 113, 0.1);
    }
    
    /* Glowing effect */
    .glow {
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from {
            box-shadow: 0 0 10px rgba(46, 204, 113, 0.4),
                        0 0 20px rgba(46, 204, 113, 0.3),
                        0 0 30px rgba(46, 204, 113, 0.2);
        }
        to {
            box-shadow: 0 0 15px rgba(46, 204, 113, 0.6),
                        0 0 25px rgba(46, 204, 113, 0.4),
                        0 0 35px rgba(46, 204, 113, 0.3);
        }
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: rgba(10, 20, 10, 0.7);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(46, 204, 113, 0.2);
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #27ae60, #2ecc71);
    }
    
    /* Custom badge */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        background: rgba(46, 204, 113, 0.2);
        border-radius: 20px;
        font-size: 0.8rem;
        color: #2ecc71;
        border: 1px solid rgba(46, 204, 113, 0.3);
    }
    
    /* Hero gradient text */
.hero-gradient {
    background: linear-gradient(135deg, #27ae60, #2ecc71);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    background-size: 200% auto;
    animation: textShine 3s ease-in-out infinite alternate;
}

/* Separate emoji from gradient text */
.hero-emoji {
    font-size: 4rem;
    margin-bottom: 10px;
    display: block;
    color: inherit !important; /* This keeps original emoji color */
}
    
    @keyframes textShine {
        to {
            background-position: 200% center;
        }
    }
    
    /* Feature icon glow */
    @keyframes iconFloat {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    </style>
    
    
    """, 
    unsafe_allow_html=True
)

# ============================================
# SIDEBAR NAVIGATION
# ============================================

st.sidebar.markdown('<div class="overlay-card">', unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <div class="sidebar-emoji">⚽</div>
    <h2 style="margin-top: 0; color: #ffffff;">Football Predictor Pro</h2>
    <div class="badge">AI-Powered Analytics</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Model status indicators
league_status = "✅" if league_model else "⚠️"
goals_status = "✅" if goals_model else "⚠️"
assists_status = "✅" if assists_model else "⚠️"
match_status = "✅" if match_model else "⚠️"

st.sidebar.markdown("### 📊 System Status")

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    st.markdown(f"""
        <div style='text-align: center;'>
            <div class='card-emoji'>🏆</div>
            <small style='color: #888;'>League</small><br>
            <b style='color: #2ecc71;'>{league_status}</b>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
        <div style='text-align: center;'>
            <div class='card-emoji'>🥅</div>
            <small style='color: #888;'>Player</small><br>
            <b style='color: #2ecc71;'>{goals_status if goals_model or assists_model else "⚠️"}</b>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
        <div style='text-align: center;'>
            <div class='card-emoji'>🎯</div>
            <small style='color: #888;'>Match</small><br>
            <b style='color: #2ecc71;'>{match_status}</b>
        </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "🔍 Navigation", 
    ["🏠 Dashboard", "🥅 Player Stats", "🎯 Match Predictor", "🏆 League Champion"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
<div style="color: #b0b0b0; font-size: 0.9rem;">
Advanced football analytics platform powered by machine learning algorithms for accurate predictions and insights.
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("</div>", unsafe_allow_html=True)

# ============================================
# PAGE: DASHBOARD
# ============================================

try:
    if "🏠 Dashboard" in page:
        # Hero Section
        st.markdown(
    """
    <div class="overlay-card" style="animation-delay: 0.1s;">
        <div style="text-align: center; margin-bottom: 30px;">
            <div class="hero-emoji">⚽</div>
            <h1 class="hero-gradient" style="font-size: 4rem; margin-bottom: 10px;">
                Football Predictor Pro
            </h1>
            <p style="font-size: 1.3rem; color: #b0b0b0; line-height: 1.6;">
                Unleash the power of AI-driven football analytics with our professional prediction platform.
            </p>
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)
        
        # Stats Counter
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                """
                <div class="overlay-card glow" style="text-align: center; animation-delay: 0.2s;">
                    <div class="dashboard-emoji">📈</div>
                    <h3>Live Predictions</h3>
                    <div style="font-size: 3rem; margin: 15px 0; color: #2ecc71;">
                        <span id="counter1">0</span>
                    </div>
                    <p style="color: #888;">Made Today</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col2:
            active_models = sum([1 if m else 0 for m in [league_model, goals_model, assists_model, match_model]])
            st.markdown(
                f"""
                <div class="overlay-card" style="text-align: center; animation-delay: 0.3s;">
                    <div class="dashboard-emoji">🤖</div>
                    <h3>AI Models</h3>
                    <div style="font-size: 3rem; margin: 15px 0; color: #2ecc71;">{active_models}</div>
                    <p style="color: #888;">Active Systems</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                """
                <div class="overlay-card" style="text-align: center; animation-delay: 0.4s;">
                    <div class="dashboard-emoji">⚡</div>
                    <h3>Processing Speed</h3>
                    <div style="font-size: 3rem; margin: 15px 0; color: #2ecc71;">0.3s</div>
                    <p style="color: #888;">Avg. Prediction Time</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # JavaScript for counter animation
        st.markdown(
            """
            <script>
            function animateCounter(elementId, target) {
                let element = document.getElementById(elementId);
                let current = 0;
                let increment = target / 50;
                let timer = setInterval(() => {
                    current += increment;
                    if (current >= target) {
                        current = target;
                        clearInterval(timer);
                    }
                    element.textContent = Math.floor(current);
                }, 30);
            }
            
            // Start counter when page loads
            window.addEventListener('load', function() {
                animateCounter('counter1', 156);
            });
            </script>
            """,
            unsafe_allow_html=True
        )
        
        # Expandable Features Section
        with st.expander("✨ Key Features", expanded=False):
            st.markdown(
                """
                <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px; flex-wrap: wrap;">
                    <div class="badge">📊 Real-time Analytics</div>
                    <div class="badge">🤖 Machine Learning</div>
                    <div class="badge">📈 Historical Data</div>
                    <div class="badge">⚡ Live Predictions</div>
                    <div class="badge">🎯 Advanced Algorithms</div>
                    <div class="badge">📱 Multi-platform</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Feature Cards
        st.markdown('<h2 style="text-align: center; margin-bottom: 30px;">🎯 Prediction Modules</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                """
                <div class="overlay-card" style="text-align: center; height: 280px; animation-delay: 0.2s;">
                    <div class="card-emoji">🏆</div>
                    <h3>League Champion</h3>
                    <p style="color: #b0b0b0; font-size: 0.9rem;">
                        Predict championship winners using comprehensive team metrics and AI analysis.
                    </p>
                    <div style="margin-top: 20px;">
                        <small style="color: #2ecc71;">Team Analytics • Season Data</small>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col2:
            player_status = "Available" if (goals_model or assists_model) else "Coming Soon"
            player_color = "#2ecc71" if (goals_model or assists_model) else "#f39c12"
            st.markdown(
                f"""
                <div class="overlay-card" style="text-align: center; height: 280px; animation-delay: 0.4s; border-left: 5px solid {player_color};">
                    <div class="card-emoji">🥅</div>
                    <h3>Player Performance</h3>
                    <p style="color: #b0b0b0; font-size: 0.9rem;">
                        Forecast player statistics and performance metrics using advanced algorithms.
                    </p>
                    <div style="margin-top: 20px;">
                        <small style="color: {player_color};">
                            {player_status} • Performance Metrics
                        </small>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                """
                <div class="overlay-card" style="text-align: center; height: 280px; animation-delay: 0.6s;">
                    <div class="card-emoji">🎯</div>
                    <h3>Match Outcomes</h3>
                    <p style="color: #b0b0b0; font-size: 0.9rem;">
                        Predict match results using team form, streaks, and head-to-head analysis.
                    </p>
                    <div style="margin-top: 20px;">
                        <small style="color: #2ecc71;">Live Analysis • Team Form</small>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # How it Works Section
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<h2 style="text-align: center; margin-bottom: 30px;">⚙️ How It Works</h2>', unsafe_allow_html=True)
        
        steps = [
            ("📊", "Data Collection", "Gather comprehensive football statistics from multiple sources"),
            ("🤖", "AI Processing", "Machine learning models analyze patterns and trends"),
            ("🔮", "Prediction", "Generate accurate forecasts based on historical data"),
            ("📈", "Insights", "Provide detailed analysis and probability assessments")
        ]
        
        cols = st.columns(4)
        for idx, (icon, title, desc) in enumerate(steps):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div class="overlay-card" style="text-align: center; height: 220px; animation-delay: {0.2 + idx*0.2}s;">
                        <div class="card-emoji">{icon}</div>
                        <h4>{title}</h4>
                        <p style="color: #b0b0b0; font-size: 0.85rem;">{desc}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

# ============================================
# PAGE: PLAYER STATS (Goals & Assists)
# ============================================

    elif "🥅 Player Stats" in page:
        st.markdown(
            """
            <div class="overlay-card" style="animation-delay: 0.1s;">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <div class="card-emoji">🥅</div>
                    <div>
                        <h2 style="margin: 0;">Player Performance Predictor</h2>
                        <p style="color: #b0b0b0; margin: 5px 0 0 0;">
                            Predict player goals and assists using advanced performance metrics
                        </p>
                    </div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Check which models are available
        available_models = []
        if goals_model is not None:
            available_models.append("Goals")
        if assists_model is not None:
            available_models.append("Assists")
        
        if not available_models:
            st.info("""
            <div class="overlay-card" style="border-left: 4px solid #f39c12;">
                <h4 style="color: #f39c12;">⚠️ Player Models Temporarily Unavailable</h4>
                <p style="color: #b0b0b0;">
                    The player prediction models are currently being updated for improved compatibility.
                    Please check back soon or use our other prediction features:
                </p>
                <ul style="color: #b0b0b0;">
                    <li>🏆 League Champion Predictor</li>
                    <li>🎯 Match Outcome Predictor</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Target selection with only available models
            options = []
            if goals_model is not None:
                options.append("⚽ Goals Prediction")
            if assists_model is not None:
                options.append("🎯 Assists Prediction")
            
            if options:
                target = st.radio(
                    "### 🎯 Select Prediction Type",
                    options,
                    horizontal=True
                )
                
                target = "Goals" if "Goals" in target else "Assists"
                
                # Get appropriate model and metadata
                if target == "Goals":
                    model = goals_model
                    meta = goals_meta_json
                    icon = "⚽"
                    color = "#2ecc71"
                else:
                    model = assists_model
                    meta = assists_meta_json
                    icon = "🎯"
                    color = "#3498db"
                
                if meta is not None and model is not None:
                    # Get features from metadata
                    features = meta.get("features", [])
                    feature_types = meta.get("feature_types", {})
                    categories_map = meta.get("categories_map", {})
                    
                    # Input form
                    st.markdown(f"### 📝 Enter Player Statistics")
                    
                    input_vals = {}
                    col1, col2 = st.columns(2)
                    
                    for i, feature in enumerate(features):
                        with col1 if i % 2 == 0 else col2:
                            ftype = feature_types.get(feature, "numeric")
                            
                            if ftype == "categorical":
                                options = categories_map.get(feature, [])
                                if options:
                                    input_vals[feature] = st.selectbox(
                                        f"**{feature}**", 
                                        options=options,
                                        help=f"Categorical feature"
                                    )
                                else:
                                    input_vals[feature] = st.text_input(
                                        f"**{feature}**", 
                                        value="",
                                        help="Enter categorical value"
                                    )
                            else:
                                input_vals[feature] = st.number_input(
                                    f"**{feature}**", 
                                    value=0.0,
                                    step=0.1,
                                    format="%.2f",
                                    help="Numeric feature"
                                )
                    
                    # Prediction button
                    if st.button(f"🚀 Predict {target}", type="primary", use_container_width=True):
                        with st.spinner(f"Analyzing player data..."):
                            try:
                                # Create input DataFrame
                                X = pd.DataFrame([input_vals], columns=features)
                                
                                # Make prediction
                                pred = model.predict(X)[0]
                                
                                # Display result with animation
                                st.markdown("---")
                                
                                # Animated result card
                                st.markdown(
                                    f"""
                                    <div class="overlay-card glow" style="text-align: center; border-left: 5px solid {color};">
                                        <div class="card-emoji">{icon}</div>
                                        <h2>Predicted {target}</h2>
                                        <div style="font-size: 4rem; font-weight: bold; margin: 20px 0; color: {color};">
                                            {float(pred):.1f}
                                        </div>
                                        <p style="color: #b0b0b0;">
                                            Expected {target.lower()} per season
                                        </p>
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                                
                            except Exception as e:
                                st.error(f"❌ Prediction failed: {str(e)}")

# ============================================
# PAGE: MATCH PREDICTOR
# ============================================

    elif "🎯 Match Predictor" in page:
        st.markdown(
            """
            <div class="overlay-card" style="animation-delay: 0.1s;">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <div class="card-emoji">🎯</div>
                    <div>
                        <h2 style="margin: 0;">Match Outcome Predictor</h2>
                        <p style="color: #b0b0b0; margin: 5px 0 0 0;">
                            Predict match winners using team form and performance metrics
                        </p>
                    </div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        if not TENSORFLOW_AVAILABLE or match_model is None or match_feature_scaler is None:
            st.info("""
            <div class="overlay-card" style="border-left: 4px solid #f39c12;">
                <h4 style="color: #f39c12;">⚠️ Match Predictor Currently Unavailable</h4>
                <p style="color: #b0b0b0;">
                    The match prediction feature requires TensorFlow which is currently being updated.
                    Please use our other prediction features:
                </p>
                <ul style="color: #b0b0b0;">
                    <li>🏆 League Champion Predictor</li>
                    <li>🥅 Player Performance Predictor</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Match features
            match_features = [
                'HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5',
                'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5',
                'HTP', 'ATP', 'HTFormPts', 'ATFormPts', 'HTGD', 'ATGD', 
                'DiffPts', 'DiffFormPts'
            ]
            
            # Create two teams section
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(
                    """
                    <div class="overlay-card" style="text-align: center;">
                        <h3 style="color: #2ecc71;">🏠 Home Team</h3>
                        <p style="color: #888; font-size: 0.9rem;">Enter home team statistics</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    """
                    <div class="overlay-card" style="text-align: center;">
                        <h3 style="color: #3498db;">✈️ Away Team</h3>
                        <p style="color: #888; font-size: 0.9rem;">Enter away team statistics</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            # Input form with team context
            match_input = {}
            col1, col2 = st.columns(2)
            
            home_features = [f for f in match_features if f.startswith('HT') or f.startswith('Diff')]
            away_features = [f for f in match_features if f.startswith('AT')]
            
            with col1:
                for feature in home_features:
                    match_input[feature] = st.number_input(
                        f"**{feature.replace('HT', 'Home ').replace('Diff', 'Difference ')}**", 
                        value=0.0,
                        step=0.1,
                        format="%.1f"
                    )
            
            with col2:
                for feature in away_features:
                    match_input[feature] = st.number_input(
                        f"**{feature.replace('AT', 'Away ')}**", 
                        value=0.0,
                        step=0.1,
                        format="%.1f"
                    )
            
            # Prediction
            if st.button("🔮 Predict Match Outcome", type="primary", use_container_width=True):
                if match_model is None or match_feature_scaler is None:
                    st.error("Match prediction models not available")
                else:
                    with st.spinner("Analyzing match data..."):
                        try:
                            # Prepare input
                            X = pd.DataFrame([match_input])[match_features]
                            X_scaled = match_feature_scaler.transform(X)
                            proba = float(match_model.predict(X_scaled)[0][0])
                            
                            # Display results
                            st.markdown("---")
                            
                            # Probability gauge
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.markdown(
                                    f"""
                                    <div class="metric-card glow">
                                        <h3>🏠 Home Win Probability</h3>
                                        <div style="width: 100%; height: 20px; background: rgba(255,255,255,0.1); 
                                                 border-radius: 10px; margin: 20px 0; overflow: hidden;">
                                            <div style="width: {proba*100}%; height: 100%; 
                                                     background: linear-gradient(90deg, #27ae60, #2ecc71); 
                                                     transition: width 2s ease;"></div>
                                        </div>
                                        <h1 style="color: #2ecc71;">
                                            {proba:.1%}
                                        </h1>
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                            
                            # Result
                            if proba > 0.5:
                                st.balloons()
                                st.success(f"## 🏆 Prediction: HOME TEAM WINS")
                                st.markdown(f"**Confidence: {proba:.1%}**")
                            else:
                                st.warning(f"## ⚠️ Prediction: AWAY TEAM WINS OR DRAW")
                                st.markdown(f"**Home win probability: {proba:.1%}**")
                                
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {str(e)}")

# ============================================
# PAGE: LEAGUE CHAMPION
# ============================================

    elif "🏆 League Champion" in page:
        st.markdown(
            """
            <div class="overlay-card" style="animation-delay: 0.1s;">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <div class="card-emoji">🏆</div>
                    <div>
                        <h2 style="margin: 0;">League Champion Predictor</h2>
                        <p style="color: #b0b0b0; margin: 5px 0 0 0;">
                            Predict championship probabilities using comprehensive team metrics
                        </p>
                    </div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        if league_model is None or league_scaler is None or league_metadata is None:
            st.error("""
            <div class="overlay-card" style="border-left: 4px solid #e74c3c;">
                <h4 style="color: #e74c3c;">⚠️ League Models Not Available</h4>
                <p style="color: #b0b0b0;">League prediction models failed to load. Please check the model files.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Get features
            features = league_metadata.get("feature_columns", [])
            opt_thresh = threshold_data.get("optimal_threshold", 0.5)
            
            # Input form
            st.markdown("### 📊 Enter Team Statistics")
            inputs = {}
            col1, col2 = st.columns(2)
            
            for i, feature in enumerate(features):
                with col1 if i % 2 == 0 else col2:
                    inputs[feature] = st.number_input(
                        f"**{feature}**", 
                        value=0.0,
                        step=0.1,
                        format="%.2f"
                    )
            
            # Prediction
            if st.button("👑 Predict Championship Chance", type="primary", use_container_width=True):
                with st.spinner("Calculating championship probability..."):
                    try:
                        # Prepare input
                        X = pd.DataFrame([inputs])[features]
                        X_scaled = league_scaler.transform(X)
                        proba = float(league_model.predict_proba(X_scaled)[:, 1][0])
                        
                        # Display results
                        st.markdown("---")
                        
                        # Championship probability
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            crown_icon = "👑" * (3 if proba > opt_thresh else 1)
                            st.markdown(
                                f"""
                                <div class="metric-card {'glow' if proba > opt_thresh else ''}" style="text-align: center;">
                                    <div class="card-emoji">{crown_icon}</div>
                                    <h3>Championship Probability</h3>
                                    <div style="font-size: 4rem; margin: 20px 0; color: #2ecc71;">
                                        {proba:.1%}
                                    </div>
                                    <p style="color: #888;">Threshold: {opt_thresh:.1%}</p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        
                        # Decision
                        if proba > opt_thresh:
                            st.balloons()
                            st.success(f"## 🎉 CHAMPION PREDICTED!")
                            st.markdown(f"**This team has a {proba:.1%} chance of winning the championship!**")
                            st.markdown("*Analysis: Strong contender based on current statistics*")
                        else:
                            st.warning(f"## 📊 UNLIKELY CHAMPION")
                            st.markdown(f"**Probability ({proba:.1%}) is below the threshold ({opt_thresh:.1%})**")
                            st.markdown("*Analysis: Needs improvement in key areas*")
                            
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please try refreshing the page or contact support.")

# ============================================
# FOOTER
# ============================================

st.markdown(
    """
    <br><br>
    <div style="text-align: center; color: #666; padding: 30px;">
        <hr style="border-color: rgba(46, 204, 113, 0.2);">
        <p style="margin-top: 20px; color: #888;">
            ⚽ Football Predictor Pro • Powered by Advanced Machine Learning •
            <span style="color: #2ecc71;" id="year"></span>
        </p>
        <p style="font-size: 0.8rem; color: #666;">
            Advanced analytics platform for football predictions and insights
        </p>
    </div>
    
    <script>
        document.getElementById('year').textContent = new Date().getFullYear();
    </script>
    """, 
    unsafe_allow_html=True
)