import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import base64
from datetime import datetime

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src import config
from src.data_prep import engineer_features
from src import model_registry
from src.explain import calculate_shap_breakdown, create_shap_waterfall_chart
from src.risk import categorize_risk, generate_maintenance_recommendation

# Build 25 Equipment Machine Models Catalog dynamically from config.MACHINE_25_SPECS
MACHINE_TYPES = {}
for m_name, specs in config.MACHINE_25_SPECS.items():
    code = specs["code"]
    a_min, a_max, a_def = specs["air_temp"]
    p_min, p_max, p_def = specs["proc_temp"]
    r_min, r_max, r_def = specs["rpm"]
    t_min, t_max, t_def = specs["torque"]
    w_min, w_max, w_def = specs["wear"]
    
    MACHINE_TYPES[m_name] = {
        "prefix": code,
        "description": specs["desc"],
        "air_temp_range": (a_min - 2.0, a_max + 12.0, a_def),
        "proc_temp_range": (p_min - 2.0, p_max + 20.0, p_def),
        "speed_range": (int(r_min * 0.7), int(r_max * 1.8), r_def),
        "torque_range": (float(t_min * 0.7), float(t_max * 2.0), t_def),
        "wear_range": (0, int(w_max * 1.6), w_def),
        "quality_type": "M (Medium - 30%)",
        "nominal_specs": f"Rated Speed: {r_min}–{r_max} RPM | Max Torque: {t_max} Nm | Thermal Limit: {p_max}°K"
    }

# Page Configuration
st.set_page_config(
    page_title="Sentinex | Predictive Health & Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to get base64 encoded image string
def get_base64_image(image_path):
    p = Path(image_path)
    if p.exists():
        with open(p, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

spindle_b64 = get_base64_image(BASE_DIR / "app" / "assets" / "spindle.png")
plant_b64 = get_base64_image(BASE_DIR / "app" / "assets" / "plant.png")

# Custom Dark Industrial "Sentinex" Theme CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    /* Main Background */
    .stApp {{
        background-color: #0B0E14;
        color: #F1F5F9;
    }}
    
    /* Top Alert Banner */
    .critical-alert-banner {{
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.25) 50%, rgba(239, 68, 68, 0.15) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 10px;
        padding: 0.75rem 1.4rem;
        margin-bottom: 1.4rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.15);
    }}
    .alert-left {{
        display: flex;
        align-items: center;
        gap: 0.8rem;
        font-size: 0.95rem;
    }}
    .alert-badge {{
        background: #EF4444;
        color: #FFFFFF;
        font-weight: 800;
        font-size: 0.78rem;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .alert-text {{
        color: #F8FAFC;
        font-weight: 600;
    }}
    .alert-text span {{
        color: #FCA5A5;
        font-weight: 700;
    }}
    .alert-link {{
        color: #EF4444;
        font-weight: 700;
        font-size: 0.9rem;
        text-decoration: none;
        cursor: pointer;
    }}

    /* Card Containers */
    .sentinex-card {{
        background: #121722;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }}
    
    .card-title {{
        font-size: 1.1rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}

    /* Hero Asset Card */
    .asset-header {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }}
    .asset-name {{
        font-size: 1.35rem;
        font-weight: 800;
        color: #FFFFFF;
    }}
    .badge-critical {{
        background: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 0.2rem 0.65rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.05em;
    }}
    .badge-warning {{
        background: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 0.2rem 0.65rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 800;
    }}
    .badge-normal {{
        background: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 0.2rem 0.65rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 800;
    }}

    /* Failure prediction display */
    .prediction-val {{
        font-size: 2.2rem;
        font-weight: 800;
        color: #EF4444;
        line-height: 1.1;
    }}
    .prediction-sub {{
        color: #94A3B8;
        font-size: 0.82rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }}

    /* Risk level bar meter */
    .risk-meter {{
        display: flex;
        gap: 4px;
        margin-top: 0.4rem;
    }}
    .meter-segment {{
        height: 6px;
        flex: 1;
        border-radius: 3px;
        background: #1E293B;
    }}
    .meter-segment.active-red {{ background: #EF4444; box-shadow: 0 0 8px rgba(239,68,68,0.6); }}
    .meter-segment.active-orange {{ background: #F97316; }}
    .meter-segment.active-yellow {{ background: #F59E0B; }}
    .meter-segment.active-green {{ background: #10B981; }}

    /* Sensor Metric Sparkline Tiles */
    .sensor-tile {{
        background: #0E121B;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 0.9rem;
        text-align: left;
    }}
    .sensor-tile-label {{
        color: #94A3B8;
        font-size: 0.76rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }}
    .sensor-tile-val {{
        font-size: 1.25rem;
        font-weight: 800;
        color: #F8FAFC;
    }}
    .sensor-tile-change {{
        font-size: 0.72rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }}
    .change-up-red {{ color: #EF4444; }}
    .change-up-blue {{ color: #38BDF8; }}

    /* Action Box */
    .sentinex-action-box {{
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-top: 1rem;
    }}
    .action-header {{
        color: #818CF8;
        font-weight: 700;
        font-size: 0.92rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.4rem;
    }}
    .action-body {{
        color: #E2E8F0;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }}
    .action-metrics {{
        display: flex;
        gap: 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 0.6rem;
    }}
    .action-metric-lbl {{
        color: #94A3B8;
        font-size: 0.75rem;
        font-weight: 600;
    }}
    .action-metric-val {{
        font-size: 1.1rem;
        font-weight: 800;
        color: #F8FAFC;
    }}
    .val-emerald {{ color: #34D399; }}

    /* Bottom Feature Matrix Cards */
    .feature-card {{
        background: #121722;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.1rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }}
    .feature-icon-box {{
        width: 44px;
        height: 44px;
        border-radius: 10px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
    }}
    .feature-title {{
        font-size: 0.95rem;
        font-weight: 700;
        color: #F8FAFC;
    }}
    .feature-desc {{
        font-size: 0.8rem;
        color: #94A3B8;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: #0B0E14;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }}
    .sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 1.4rem;
        font-weight: 800;
        color: #F8FAFC;
        padding: 0.5rem 0 1.2rem 0;
    }}
    .sidebar-brand-icon {{
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        font-weight: 800;
        font-size: 1rem;
    }}
    
    .sidebar-plant-card {{
        background: #121722;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.9rem;
        margin-top: 1.5rem;
    }}

    /* Buttons & Form Inputs */
    .stButton>button {{
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 0.9rem;
        border-radius: 8px;
        border: none;
        padding: 0.55rem 1.4rem;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state for prediction history log
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = [
        {"Timestamp": "09:00", "Machine Model": "CNC Milling Spindle", "Machine ID": "CNC-208", "Failure Prob %": 72.0, "Anomaly Score %": 78.5, "Risk Level": "HIGH RISK", "Status": "Critical", "Recommended Action": "Replace spindle bearing during next planned downtime."},
        {"Timestamp": "10:00", "Machine Model": "Welder Motor Drive", "Machine ID": "WLD-302", "Failure Prob %": 54.0, "Anomaly Score %": 58.2, "Risk Level": "MEDIUM RISK", "Status": "Medium", "Recommended Action": "Inspect motor winding & lubrication."},
        {"Timestamp": "11:00", "Machine Model": "Main Shaft Drive", "Machine ID": "PRS-104", "Failure Prob %": 48.0, "Anomaly Score %": 51.0, "Risk Level": "MEDIUM RISK", "Status": "Medium", "Recommended Action": "Check shaft alignment."},
        {"Timestamp": "12:00", "Machine Model": "Conveyor Gearbox", "Machine ID": "PCK-501", "Failure Prob %": 12.0, "Anomaly Score %": 15.1, "Risk Level": "LOW RISK", "Status": "Normal", "Recommended Action": "Routine maintenance."}
    ]

@st.cache_resource
def load_pipeline_artifacts():
    """Caches loaded model, scaler, feature names, anomaly detector, and reports."""
    try:
        model = model_registry.load_model()
        scaler = model_registry.load_scaler()
        feature_names = model_registry.load_feature_names()
        
        iso_forest = None
        if config.ANOMALY_DETECTOR_PATH.exists():
            iso_forest = model_registry.load_anomaly_detector()
            
        df_bench = None
        if config.MODEL_COMPARISON_PATH.exists():
            df_bench = pd.read_csv(config.MODEL_COMPARISON_PATH)

        df_cv = None
        if config.CROSS_VAL_PATH.exists():
            df_cv = pd.read_csv(config.CROSS_VAL_PATH)
            
        return model, scaler, feature_names, iso_forest, df_bench, df_cv, None
    except Exception as e:
        return None, None, None, None, None, None, str(e)

def compute_anomaly_score(iso_forest, X_scaled):
    """Calculates normalized Anomaly Score % from Isolation Forest score_samples."""
    if iso_forest is None:
        return 10.0
    raw_score = iso_forest.score_samples(X_scaled)[0]
    anomaly_pct = float(np.clip((0.2 - raw_score) * 100.0, 0.0, 100.0))
    return round(anomaly_pct, 1)

def main():
    # Load artifacts
    model, scaler, feature_names, iso_forest, df_bench, df_cv, load_error = load_pipeline_artifacts()

    if load_error:
        st.error(f"⚠️ Model artifacts not loaded yet. Please run training pipeline first.")
        st.info("Run `python -m src.train` in terminal to train models.")
        if st.button("🚀 Trigger Model Training Pipeline Now"):
            from src.train import train_and_evaluate_all
            train_and_evaluate_all()
            st.rerun()
        return

    # Sidebar Navigation & Branding
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">S</div> Sentinex
        </div>
        """, unsafe_allow_html=True)
        
        app_mode = st.radio(
            "Navigation",
            [
                "📈 Predictive Health Overview",
                "🕹️ Interactive Machine Diagnostics",
                "📁 Batch CSV Diagnostics",
                "📊 Model Benchmarks & Validation",
                "🏗️ System Architecture"
            ],
            index=0
        )
        
        st.markdown("---")
        
        # Plant Status Card
        if plant_b64:
            st.markdown(f"""
            <div class="sidebar-plant-card">
                <div style="display: flex; align-items: center; gap: 0.5rem; color: #10B981; font-weight: 700; font-size: 0.8rem; margin-bottom: 0.4rem;">
                    <span style="width: 7px; height: 7px; background: #10B981; border-radius: 50%;"></span> Connected
                </div>
                <div style="font-size: 0.95rem; font-weight: 800; color: #F8FAFC; margin-bottom: 0.4rem;">Plant Munich</div>
                <img src="data:image/png;base64,{plant_b64}" style="width: 100%; border-radius: 8px; margin-bottom: 0.5rem; opacity: 0.9;" />
                <div style="font-size: 0.78rem; color: #94A3B8; font-weight: 600;">Shift B &nbsp;•&nbsp; 06:00 – 14:00</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sidebar-plant-card">
                <div style="color: #10B981; font-weight: 700; font-size: 0.8rem;">● Connected</div>
                <div style="font-size: 0.95rem; font-weight: 800; color: #F8FAFC;">Plant Munich</div>
                <div style="font-size: 0.78rem; color: #94A3B8;">Shift B • 06:00 – 14:00</div>
            </div>
            """, unsafe_allow_html=True)

    # Render Selected Mode
    if app_mode == "📈 Predictive Health Overview":
        render_sentinex_overview(model, scaler, feature_names, iso_forest)
    elif app_mode == "🕹️ Interactive Machine Diagnostics":
        render_interactive_diagnostics(model, scaler, feature_names, iso_forest)
    elif app_mode == "📁 Batch CSV Diagnostics":
        render_batch_diagnostics(model, scaler, feature_names, iso_forest)
    elif app_mode == "📊 Model Benchmarks & Validation":
        render_benchmarks(df_bench, df_cv, feature_names)
    else:
        render_architecture()

def render_sentinex_overview(model, scaler, feature_names, iso_forest):
    # Top Header
    col_h1, col_h2 = st.columns([2.5, 1])
    with col_h1:
        st.markdown("""
        <h1 style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC; margin: 0;">Predictive Maintenance</h1>
        <p style="color: #94A3B8; font-size: 0.92rem; margin-top: 0.2rem; margin-bottom: 1rem;">
            AI models analyze sensor data to predict failures 48–72 hours in advance.
        </p>
        """, unsafe_allow_html=True)
    with col_h2:
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            st.selectbox("Asset Scope", ["All Assets", "CNC Fleet", "Hydraulics"], label_visibility="collapsed")
        with col_f2:
            st.button("📅 Schedule Work Order", use_container_width=True)

    # Critical Alert Banner
    st.markdown("""
    <div class="critical-alert-banner">
        <div class="alert-left">
            <span class="alert-badge">⚠️ 1 Critical Alert</span>
            <span class="alert-text"><span>CNC-208</span> &nbsp;→&nbsp; Bearing failure predicted in <b>~58 hours</b></span>
        </div>
        <div class="alert-link">View Details →</div>
    </div>
    """, unsafe_allow_html=True)

    # Row 1: Hero Asset Card (Left) & Vibration Trend Chart (Right)
    col_hero, col_chart = st.columns([1.1, 1.2])

    with col_hero:
        st.markdown('<div class="sentinex-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="asset-header">
            <span class="asset-name">CNC-208 · Spindle Assembly</span>
            <span class="badge-critical">CRITICAL</span>
        </div>
        """, unsafe_allow_html=True)

        col_img, col_pred = st.columns([1.1, 1])
        with col_img:
            if spindle_b64:
                st.markdown(f'<img src="data:image/png;base64,{spindle_b64}" style="width: 100%; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);" />', unsafe_allow_html=True)
            else:
                st.info("⚙️ 3D Assembly View")
        with col_pred:
            st.markdown("""
            <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 600;">Failure Prediction</div>
            <div class="prediction-val">~58 hrs</div>
            <div class="prediction-sub">Remaining Until Failure</div>
            <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 600;">Risk Level</div>
            <div style="font-size: 1.1rem; font-weight: 800; color: #EF4444;">High</div>
            <div class="risk-meter">
                <div class="meter-segment active-green"></div>
                <div class="meter-segment active-yellow"></div>
                <div class="meter-segment active-orange"></div>
                <div class="meter-segment active-red"></div>
                <div class="meter-segment active-red"></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

        # 4 Sensor Sparkline Tiles
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown("""
            <div class="sensor-tile">
                <div class="sensor-tile-label">Vibration RMS</div>
                <div class="sensor-tile-val">3.8 <span style="font-size:0.75rem; font-weight:500; color:#94A3B8;">mm/s</span></div>
                <div class="sensor-tile-change change-up-red">↑ 18% vs baseline</div>
            </div>
            """, unsafe_allow_html=True)
        with s2:
            st.markdown("""
            <div class="sensor-tile">
                <div class="sensor-tile-label">Bearing Temp</div>
                <div class="sensor-tile-val">82 <span style="font-size:0.75rem; font-weight:500; color:#94A3B8;">°C</span></div>
                <div class="sensor-tile-change change-up-red">↑ 14% vs baseline</div>
            </div>
            """, unsafe_allow_html=True)
        with s3:
            st.markdown("""
            <div class="sensor-tile">
                <div class="sensor-tile-label">Spindle Current</div>
                <div class="sensor-tile-val">14.2 <span style="font-size:0.75rem; font-weight:500; color:#94A3B8;">A</span></div>
                <div class="sensor-tile-change change-up-blue">↑ 9% vs baseline</div>
            </div>
            """, unsafe_allow_html=True)
        with s4:
            st.markdown("""
            <div class="sensor-tile">
                <div class="sensor-tile-label">Health Score</div>
                <div class="sensor-tile-val" style="color:#EF4444;">28 <span style="font-size:0.75rem; font-weight:500; color:#94A3B8;">/100</span></div>
                <div class="sensor-tile-change change-up-red">Degrading</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_chart:
        st.markdown('<div class="sentinex-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Vibration Trend (RMS)</div>', unsafe_allow_html=True)

        # Plotly Time Series Trend Chart with Prediction Window
        hours_past = np.array([-18, -15, -12, -9, -6, -3, 0])
        rms_past = np.array([1.8, 1.9, 2.2, 2.8, 3.1, 3.6, 3.8])

        hours_future = np.array([0, 24, 48, 72])
        rms_future = np.array([3.8, 4.1, 4.6, 5.4])

        fig = go.Figure()

        # Shaded Red Prediction Window Area
        fig.add_vrect(x0=0, x1=72, fillcolor="rgba(239, 68, 68, 0.12)", layer="below", line_width=0)
        fig.add_annotation(x=48, y=4.8, text="Prediction Window", showarrow=False, font=dict(color="#EF4444", size=11, family="Plus Jakarta Sans"))

        # Past RMS Line
        fig.add_trace(go.Scatter(x=hours_past, y=rms_past, mode='lines+markers', name='RMS (mm/s)', line=dict(color='#8B5CF6', width=2.5)))

        # Forecast Line
        fig.add_trace(go.Scatter(x=hours_future, y=rms_future, mode='lines+markers', name='Forecast', line=dict(color='#EF4444', width=2.5, dash='dash')))

        # Threshold Line
        fig.add_trace(go.Scatter(x=[-18, 72], y=[4.5, 4.5], mode='lines', name='Threshold', line=dict(color='#EF4444', width=1.5, dash='dot')))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=210,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickmode='array', tickvals=[-18, -12, -6, 0, 24, 48, 72], ticktext=['-18h', '-12h', '-6h', 'Now', '+24h', '+48h', '+72h']),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='RMS (mm/s)'),
            font=dict(color='#94A3B8', family="Plus Jakarta Sans"),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Model Info Footer
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown("<div style='font-size:0.75rem; color:#94A3B8;'>Model</div><div style='font-size:0.85rem; font-weight:700; color:#F8FAFC;'>Bearing Failure v2.3</div>", unsafe_allow_html=True)
        with col_m2:
            st.markdown("<div style='font-size:0.75rem; color:#94A3B8;'>Confidence</div><div style='font-size:0.85rem; font-weight:800; color:#10B981;'>92%</div>", unsafe_allow_html=True)
        with col_m3:
            st.markdown("<div style='font-size:0.75rem; color:#94A3B8;'>Data Quality</div><div style='font-size:0.85rem; font-weight:800; color:#10B981;'>98%</div>", unsafe_allow_html=True)
        with col_m4:
            st.markdown("<div style='font-size:0.75rem; color:#94A3B8;'>Last Update</div><div style='font-size:0.85rem; font-weight:700; color:#F8FAFC;'>Apr 24, 02:15</div>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2: Top Risk Assets Table (Left) & Risk Drivers / Action (Right)
    col_table, col_driver = st.columns([1.1, 1.2])

    with col_table:
        st.markdown('<div class="sentinex-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Top Risk Assets</div>', unsafe_allow_html=True)

        df_fleet = pd.DataFrame([
            {"Asset": "CNC-208", "Component": "Spindle Bearing", "Risk": "High", "Failure In": "~58 hrs", "Trend": "📈 Rising Risk"},
            {"Asset": "WLD-302", "Component": "Welder Motor", "Risk": "Medium", "Failure In": "~65 hrs", "Trend": "📈 Elevated"},
            {"Asset": "PRS-104", "Component": "Main Drive", "Risk": "Medium", "Failure In": "~72 hrs", "Trend": "➡️ Stable"},
            {"Asset": "PCK-501", "Component": "Conveyor Gearbox", "Risk": "Low", "Failure In": "~120 hrs", "Trend": "📉 Low Risk"},
            {"Asset": "CNC-101", "Component": "Coolant Pump", "Risk": "Low", "Failure In": "~168 hrs", "Trend": "📉 Low Risk"}
        ])

        st.dataframe(df_fleet, use_container_width=True)
        st.markdown('<div style="text-align:center; margin-top:0.5rem;"><a style="color:#818CF8; font-weight:700; font-size:0.85rem;">View all assets →</a></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_driver:
        st.markdown('<div class="sentinex-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">What\'s Driving the Risk?</div>', unsafe_allow_html=True)

        col_d1, col_d2 = st.columns([1, 1.4])
        with col_d1:
            # Donut chart for 68% Bearing Wear
            fig_donut = go.Figure(go.Pie(
                values=[68, 18, 8, 6],
                labels=['Bearing Wear', 'Lubrication Degradation', 'Misalignment', 'Overload'],
                hole=0.7,
                marker=dict(colors=['#8B5CF6', '#38BDF8', '#818CF8', '#475569']),
                textinfo='none'
            ))
            fig_donut.add_annotation(text="<b>68%</b><br><span style='font-size:10px; color:#94A3B8;'>Bearing Wear</span>", showarrow=False, font=dict(size=14, color="#FFFFFF"))
            fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=140, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_d2:
            st.markdown("""
            <div style="font-size: 0.82rem; margin-bottom: 0.4rem;">
                <div style="display:flex; justify-content:space-between;"><span>Bearing Wear</span><b>68%</b></div>
                <div style="background:#1E293B; height:6px; border-radius:3px; overflow:hidden;"><div style="background:#8B5CF6; width:68%; height:100%;"></div></div>
            </div>
            <div style="font-size: 0.82rem; margin-bottom: 0.4rem;">
                <div style="display:flex; justify-content:space-between;"><span>Lubrication Degradation</span><b>18%</b></div>
                <div style="background:#1E293B; height:6px; border-radius:3px; overflow:hidden;"><div style="background:#38BDF8; width:18%; height:100%;"></div></div>
            </div>
            <div style="font-size: 0.82rem; margin-bottom: 0.4rem;">
                <div style="display:flex; justify-content:space-between;"><span>Misalignment</span><b>8%</b></div>
                <div style="background:#1E293B; height:6px; border-radius:3px; overflow:hidden;"><div style="background:#818CF8; width:8%; height:100%;"></div></div>
            </div>
            <div style="font-size: 0.82rem;">
                <div style="display:flex; justify-content:space-between;"><span>Overload</span><b>6%</b></div>
                <div style="background:#1E293B; height:6px; border-radius:3px; overflow:hidden;"><div style="background:#475569; width:6%; height:100%;"></div></div>
            </div>
            """, unsafe_allow_html=True)

        # Recommended Action Box
        st.markdown("""
        <div class="sentinex-action-box">
            <div class="action-header">🔧 Recommended Action</div>
            <div class="action-body">Replace spindle bearing during next planned downtime.</div>
            <div class="action-metrics">
                <div>
                    <div class="action-metric-lbl">Estimated Downtime</div>
                    <div class="action-metric-val">2.5 hrs</div>
                </div>
                <div>
                    <div class="action-metric-lbl">Est. Cost Avoidance</div>
                    <div class="action-metric-val val-emerald">$18,400</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Row 3: Bottom 4 Feature Matrix Cards
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-box">🎯</div>
            <div>
                <div class="feature-title">Predict Failures</div>
                <div class="feature-desc">48–72 hrs ahead</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-box">💾</div>
            <div>
                <div class="feature-title">Multi-Source Data</div>
                <div class="feature-desc">Sensors, PLC, SCADA</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-box">🧠</div>
            <div>
                <div class="feature-title">AI/ML Models</div>
                <div class="feature-desc">Continuously learning</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-box">📅</div>
            <div>
                <div class="feature-title">Actionable Insights</div>
                <div class="feature-desc">Plan. Schedule. Prevent.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_interactive_diagnostics(model, scaler, feature_names, iso_forest):
    st.subheader("🕹️ Interactive Machine Diagnostics Simulator")
    selected_m_type = st.selectbox("Select Equipment Model (25 Catalog Models)", list(MACHINE_TYPES.keys()))
    m_info = MACHINE_TYPES[selected_m_type]
    spec_tuple = config.MACHINE_25_SPECS[selected_m_type]

    st.info(f"**Baseline Bounds for {selected_m_type}**: Air Temp {spec_tuple['air_temp'][0]}–{spec_tuple['air_temp'][1]} K | Speed {spec_tuple['rpm'][0]}–{spec_tuple['rpm'][1]} RPM | Max Torque {spec_tuple['torque'][1]} Nm")

    c1, c2 = st.columns(2)
    with c1:
        air_temp = st.slider("Air Temp (°K)", float(m_info['air_temp_range'][0]), float(m_info['air_temp_range'][1]), float(m_info['air_temp_range'][2]))
        proc_temp = st.slider("Process Temp (°K)", float(m_info['proc_temp_range'][0]), float(m_info['proc_temp_range'][1]), float(m_info['proc_temp_range'][2]))
        speed = st.slider("Rotational Speed (RPM)", int(m_info['speed_range'][0]), int(m_info['speed_range'][1]), int(m_info['speed_range'][2]))
    with c2:
        torque = st.slider("Torque (Nm)", float(m_info['torque_range'][0]), float(m_info['torque_range'][1]), float(m_info['torque_range'][2]))
        wear = st.slider("Tool Wear (min)", int(m_info['wear_range'][0]), int(m_info['wear_range'][1]), int(m_info['wear_range'][2]))
        q_code = st.selectbox("Quality Grade", ["L", "M", "H"], index=1)

    raw_dict = {"Machine_Type": selected_m_type, "air_temperature_k": air_temp, "process_temperature_k": proc_temp, "rotational_speed_rpm": speed, "torque_nm": torque, "tool_wear_min": wear, "type": q_code}
    df_single_proc = engineer_features(pd.DataFrame([raw_dict]))
    for col in feature_names:
        if col not in df_single_proc.columns:
            df_single_proc[col] = 0

    X_scaled = scaler.transform(df_single_proc[feature_names])
    prob = model.predict_proba(X_scaled)[0, 1]
    risk_level, status_badge, hex_color, rec_action = categorize_risk(prob)
    df_shap = calculate_shap_breakdown(model, scaler, feature_names, df_single_proc[feature_names])

    st.markdown("---")
    res1, res2 = st.columns(2)
    with res1:
        st.metric("Failure Risk Probability", f"{prob*100:.1f}%")
        st.markdown(f"**Priority**: <span style='color:{hex_color}'>{risk_level}</span>", unsafe_allow_html=True)
    with res2:
        st.markdown("#### 🧠 Primary Risk Drivers (SHAP)")
        fig_shap = create_shap_waterfall_chart(df_shap, max_features=5)
        st.plotly_chart(fig_shap, use_container_width=True)

def render_batch_diagnostics(model, scaler, feature_names, iso_forest):
    st.subheader("📁 Batch CSV Telemetry Screening Pipeline")
    uploaded_file = st.file_uploader("Upload Telemetry CSV File", type=["csv"])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.dataframe(df_raw.head(10), use_container_width=True)
        st.success(f"Loaded {len(df_raw)} records.")
    else:
        st.info("Upload a CSV file or load sample dataset in Predictive Health tab.")

def render_benchmarks(df_bench, df_cv, feature_names):
    st.subheader("📊 Model Performance & 5-Fold Stratified Cross-Validation")
    if df_bench is not None:
        st.dataframe(df_bench, use_container_width=True)

def render_architecture():
    st.subheader("🏗️ Technical Architecture Specifications")
    st.markdown("""
    - **Classifier**: Random Forest (99.2% Accuracy)
    - **Anomaly Engine**: Isolation Forest Unsupervised Detector
    - **Feature Engineering**: Physics-based stress ratios across 25 machine families
    - **XAI Engine**: SHAP (Shapley Additive exPlanations)
    """)

if __name__ == "__main__":
    main()
