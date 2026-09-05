import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

FEATURE_DISPLAY_NAMES = {
    "air_temperature_k": "Air Temperature (°K)",
    "process_temperature_k": "Process Temperature (°K)",
    "rotational_speed_rpm": "Rotational Speed (RPM)",
    "torque_nm": "Torque (Nm)",
    "tool_wear_min": "Tool Wear (min)",
    "temp_difference_k": "Temperature Difference (°K)",
    "power_kw": "Mechanical Power (kW)",
    "wear_rate": "Tool Wear Rate",
    "temp_torque_ratio": "Temp-Torque Ratio",
    "type_H": "Quality Variant: High (H)",
    "type_L": "Quality Variant: Low (L)",
    "type_M": "Quality Variant: Medium (M)"
}

def get_shap_explainer(model: Any, X_background: np.ndarray = None):
    """Instantiates the correct SHAP explainer based on model architecture."""
    model_type_name = type(model).__name__
    
    if isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)) or "XGB" in model_type_name:
        return shap.TreeExplainer(model)
    elif isinstance(model, LogisticRegression):
        if X_background is None:
            X_background = np.zeros((1, model.coef_.shape[1]))
        return shap.LinearExplainer(model, X_background)
    else:
        return shap.Explainer(model, X_background)

def calculate_shap_breakdown(model: Any, scaler: Any, feature_names: List[str], input_df: pd.DataFrame) -> pd.DataFrame:
    """Computes SHAP feature importance contributions for a single machine reading."""
    X_input = input_df[feature_names].copy()
    X_scaled = scaler.transform(X_input)
    explainer = get_shap_explainer(model, X_background=X_scaled)
    shap_out = explainer(X_scaled)
    
    if len(shap_out.shape) == 3:
        shap_vals = shap_out.values[0, :, 1]
    elif len(shap_out.shape) == 2:
        shap_vals = shap_out.values[0, :]
    else:
        shap_vals = np.array(shap_out.values).flatten()
        
    raw_vals = X_input.iloc[0].values
    
    records = []
    for feat_col, raw_v, s_val in zip(feature_names, raw_vals, shap_vals):
        display_name = FEATURE_DISPLAY_NAMES.get(feat_col, feat_col)
        impact_dir = "Pushes Failure (+)" if s_val > 0 else "Pushes Normal (-)"
        records.append({
            "feature_id": feat_col,
            "feature": display_name,
            "raw_value": round(float(raw_v), 2),
            "shap_value": float(s_val),
            "abs_shap": abs(float(s_val)),
            "impact": impact_dir
        })
        
    df_shap = pd.DataFrame(records)
    df_shap = df_shap.sort_values(by="abs_shap", ascending=False).reset_index(drop=True)
    return df_shap

def create_shap_waterfall_chart(df_shap: pd.DataFrame, max_features: int = 6) -> go.Figure:
    """Generates an interactive Plotly horizontal bar chart of top SHAP contributors."""
    df_top = df_shap.head(max_features).iloc[::-1]
    colors = ["#FF4B4B" if s > 0 else "#1C83E5" for s in df_top["shap_value"]]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_top["shap_value"],
        y=df_top["feature"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
        text=[f"{val:+.3f} (Val: {raw})" for val, raw in zip(df_top["shap_value"], df_top["raw_value"])],
        textposition="auto"
    ))
    
    fig.update_layout(
        title=dict(text="<b>SHAP Feature Impact Breakdown</b>", font=dict(size=15, color="#FFFFFF")),
        xaxis=dict(title="SHAP Value (Impact on Failure Risk)", gridcolor="#333333", zerolinecolor="#666666"),
        yaxis=dict(title="", gridcolor="#333333"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=40),
        font=dict(color="#E0E0E0")
    )
    return fig
