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
    Generates actionable maintenance advice by mapping top positive SHAP failure drivers
    to physical maintenance protocols.
    """
    if risk_level == "LOW RISK":
        return {
            "headline": "🟢 CONTINUE NORMAL OPERATION",
            "action": "Continue normal operation",
            "details": [
                "All sensor parameters within normal operational baseline.",
                "Routine preventive maintenance according to standard schedule."
            ]
        }
        
    # Find positive SHAP contributors (pushing toward failure)
    positive_drivers = df_shap[df_shap["shap_value"] > 0]
    top_driver_ids = positive_drivers.head(3)["feature_id"].tolist() if not positive_drivers.empty else []
    
    has_torque = any("torque" in f for f in top_driver_ids)
    has_temp = any("temp" in f for f in top_driver_ids)
    has_wear = any("wear" in f for f in top_driver_ids)
    has_speed = any("speed" in f or "rpm" in f for f in top_driver_ids)
    
    actions = []
    
    # Combined High Torque + Temperature special condition
    if has_torque and has_temp:
        actions.append("⚠️ **Combined High Torque + Temperature**: Prioritize immediate mechanical inspection of drive and cooling systems.")
    else:
        if has_torque:
            actions.append("🔧 **High Torque Detected**: Inspect drive/load system and spindle bearings for mechanical strain.")
        if has_temp:
            actions.append("🌡️ **High Temperature Detected**: Inspect cooling and lubrication system, heat exchanger, and fluid lines.")
            
    if has_wear:
        actions.append("🛠️ **Excessive Tool Wear**: Inspect or replace tooling inserts/assembly immediately.")
        
    if has_speed:
        actions.append("🔄 **Over-Speed RPM**: Calibrate rotational speed regulator and motor controller.")
        
    if not actions:
        actions.append("🛠️ **General Diagnostic**: Conduct comprehensive mechanical and electrical diagnostic scan.")
        
    if risk_level == "CRITICAL RISK":
        headline = "🔴 CRITICAL RISK — IMMEDIATE MAINTENANCE REQUIRED"
    elif risk_level == "HIGH RISK":
        headline = "🟠 HIGH RISK — SCHEDULE INSPECTION PROMPTLY"
    else:
        headline = "🟡 MEDIUM RISK — MONITOR MACHINE CLOSELY"
        
    top_driver_names = positive_drivers.head(2)["feature"].tolist() if not positive_drivers.empty else ["Multivariate Sensor Drift"]
    
    return {
        "headline": headline,
        "action": f"Top Failure Drivers: {', '.join(top_driver_names)}",
        "details": actions
    }

