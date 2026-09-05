import pandas as pd
from typing import Dict, Any, List, Tuple

def categorize_risk(failure_prob: float) -> Tuple[str, str, str, str]:
    """
    Categorizes failure probability into 4 risk bands.
    Returns: (risk_level, status_badge, hex_color, recommended_action)
    """
    prob_pct = failure_prob * 100.0
    
    if failure_prob >= 0.80:
        return (
            "CRITICAL RISK",
            f"🔴 CRITICAL RISK ({prob_pct:.1f}%)",
            "#FF4B4B",
            "Immediate Maintenance Required"
        )
    elif failure_prob >= 0.60:
        return (
            "HIGH RISK",
            f"🟠 HIGH RISK ({prob_pct:.1f}%)",
            "#FF7A00",
            "Schedule Inspection Promptly"
        )
    elif failure_prob >= 0.30:
        return (
            "MEDIUM RISK",
            f"🟡 MEDIUM RISK ({prob_pct:.1f}%)",
            "#FFD166",
            "Monitor Machine Closely"
        )
    else:
        return (
            "LOW RISK",
            f"🟢 NORMAL OPERATION ({prob_pct:.1f}%)",
            "#00CC96",
            "Continue Normal Operation"
        )

def generate_maintenance_recommendation(risk_level: str, df_shap: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates actionable maintenance advice based on top positive SHAP failure drivers.
    """
    if risk_level == "LOW RISK":
        return {
            "headline": "🟢 CONTINUE NORMAL OPERATION",
            "action": "Routine preventive maintenance according to standard schedule.",
            "details": ["All sensor parameters within normal operational bounds.", "No technician intervention required."]
        }
        
    # Find positive SHAP contributors (pushing toward failure)
    positive_drivers = df_shap[df_shap["shap_value"] > 0]
    
    actions = []
    if not positive_drivers.empty:
        top_driver_ids = positive_drivers.head(3)["feature_id"].tolist()
        
        for feat in top_driver_ids:
            if "temp" in feat:
                actions.append("🌡️ **Cooling System**: Inspect process heat exchanger, check coolant flow rate, and clear ventilation filters.")
            elif "torque" in feat:
                actions.append("⚙️ **Drive Motor & Spindle**: Check mechanical torque limits, inspect drive belt tension and spindle bearings.")
            elif "wear" in feat:
                actions.append("🔧 **Cutting Tool Assembly**: High cumulative tool wear detected. Schedule tool insert/bit replacement.")
            elif "speed" in feat:
                actions.append("🔄 **RPM Regulator**: Calibrate motor drive controller and check for rotational speed oscillation.")
            elif "power" in feat:
                actions.append("⚡ **Power Supply**: Inspect motor electrical draw and verify power transmission efficiency.")
                
    if not actions:
        actions.append("🛠️ Conduct comprehensive mechanical and electrical diagnostic scan.")
        
    if risk_level == "CRITICAL RISK":
        headline = "🔴 IMMEDIATE MAINTENANCE REQUIRED"
    elif risk_level == "HIGH RISK":
        headline = "🟠 SCHEDULE INSPECTION PROMPTLY"
    else:
        headline = "🟡 MONITOR MACHINE CLOSELY"
        
    return {
        "headline": headline,
        "action": f"Top anomaly drivers: {', '.join(positive_drivers.head(2)['feature'].tolist()) if not positive_drivers.empty else 'Multivariate Sensor Drift'}",
        "details": actions
    }
