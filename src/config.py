import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure directories exist
for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# UCI AI4I 2020 Dataset Direct URL
UCI_DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
RAW_DATA_PATH = RAW_DATA_DIR / "ai4i2020.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_features.csv"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"
ANOMALY_DETECTOR_PATH = MODELS_DIR / "anomaly_detector.joblib"
CROSS_VAL_PATH = REPORTS_DIR / "cross_validation.csv"
TARGET_COL = "target"


# UCI Dataset Original Column Mapping
COLUMN_MAPPING = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "target"
}

# Master Specs for 25 Industrial Machine Types (Normal Baseline Ranges)
MACHINE_25_SPECS = {
    "CNC Milling Spindle (High Precision)": {
        "code": "M-CNC",
        "air_temp": (296.0, 303.0, 299.5),
        "proc_temp": (304.0, 313.0, 308.5),
        "rpm": (1200, 3000, 2100),
        "torque": (25.0, 65.0, 45.0),
        "wear": (10, 180, 75),
        "desc": "High-precision spindle for aerospace & automotive milling."
    },
    "Hydraulic Drive Pump (High Torque)": {
        "code": "M-HYD",
        "air_temp": (298.0, 306.0, 302.0),
        "proc_temp": (308.0, 318.0, 313.0),
        "rpm": (900, 1800, 1350),
        "torque": (45.0, 100.0, 72.5),
        "wear": (20, 220, 110),
        "desc": "Heavy-duty hydraulic power unit operating under high load."
    },
    "Industrial Gas Turbine Compressor": {
        "code": "M-GAS",
        "air_temp": (295.0, 305.0, 300.0),
        "proc_temp": (315.0, 335.0, 325.0),
        "rpm": (3000, 6000, 4500),
        "torque": (80.0, 180.0, 130.0),
        "wear": (50, 300, 150),
        "desc": "High-velocity gas compression turbine operating at high RPM."
    },
    "Conveyor Drive Gearbox (Heavy Load)": {
        "code": "M-CONV",
        "air_temp": (297.0, 307.0, 302.0),
        "proc_temp": (307.0, 320.0, 313.5),
        "rpm": (700, 1500, 1100),
        "torque": (60.0, 140.0, 100.0),
        "wear": (30, 250, 120),
        "desc": "Continuous production line conveyor main drive."
    },
    "Robotic Arm Servo Joint Actuator": {
        "code": "M-ROB",
        "air_temp": (296.0, 302.0, 299.0),
        "proc_temp": (304.0, 312.0, 308.0),
        "rpm": (1500, 4000, 2750),
        "torque": (15.0, 50.0, 32.5),
        "wear": (10, 160, 60),
        "desc": "Multi-axis servo joint actuator used in automated assembly."
    },
    "Centrifugal Water Pump": {
        "code": "M-PUMP",
        "air_temp": (295.0, 303.0, 299.0),
        "proc_temp": (304.0, 314.0, 309.0),
        "rpm": (1200, 2800, 2000),
        "torque": (20.0, 70.0, 45.0),
        "wear": (20, 200, 90),
        "desc": "High-volume centrifugal fluid circulation pump."
    },
    "Industrial Air Compressor": {
        "code": "M-COMP",
        "air_temp": (297.0, 306.0, 301.5),
        "proc_temp": (310.0, 325.0, 317.5),
        "rpm": (1000, 2500, 1750),
        "torque": (40.0, 110.0, 75.0),
        "wear": (20, 240, 110),
        "desc": "Pneumatic supply multi-stage air compressor."
    },
    "Industrial Cooling Fan": {
        "code": "M-FAN",
        "air_temp": (294.0, 302.0, 298.0),
        "proc_temp": (301.0, 310.0, 305.5),
        "rpm": (900, 2200, 1550),
        "torque": (10.0, 35.0, 22.5),
        "wear": (10, 180, 80),
        "desc": "HVAC & heat exchanger forced-air cooling fan."
    },
    "CNC Lathe Spindle": {
        "code": "M-LATHE",
        "air_temp": (296.0, 303.0, 299.5),
        "proc_temp": (305.0, 315.0, 310.0),
        "rpm": (1500, 4000, 2750),
        "torque": (20.0, 70.0, 45.0),
        "wear": (15, 200, 85),
        "desc": "Rotational turning lathe headstock drive spindle."
    },
    "CNC Drilling Spindle": {
        "code": "M-DRILL",
        "air_temp": (296.0, 304.0, 300.0),
        "proc_temp": (305.0, 316.0, 310.5),
        "rpm": (1800, 4500, 3150),
        "torque": (15.0, 55.0, 35.0),
        "wear": (15, 190, 80),
        "desc": "High-RPM hole drilling precision spindle unit."
    },
    "Injection Molding Drive": {
        "code": "M-INJ",
        "air_temp": (298.0, 307.0, 302.5),
        "proc_temp": (315.0, 340.0, 327.5),
        "rpm": (500, 1800, 1150),
        "torque": (50.0, 130.0, 90.0),
        "wear": (30, 250, 120),
        "desc": "High-pressure plastic injection hydraulic drive."
    },
    "Plastic Extruder Drive": {
        "code": "M-EXT",
        "air_temp": (298.0, 308.0, 303.0),
        "proc_temp": (320.0, 350.0, 335.0),
        "rpm": (400, 1500, 950),
        "torque": (60.0, 150.0, 105.0),
        "wear": (30, 280, 130),
        "desc": "Continuous polymer extrusion screw motor drive."
    },
    "Industrial Blower": {
        "code": "M-BLOW",
        "air_temp": (295.0, 304.0, 299.5),
        "proc_temp": (304.0, 315.0, 309.5),
        "rpm": (1000, 2500, 1750),
        "torque": (15.0, 45.0, 30.0),
        "wear": (20, 200, 90),
        "desc": "Positive displacement air & gas handling blower."
    },
    "Mixing / Agitator Motor": {
        "code": "M-MIX",
        "air_temp": (297.0, 306.0, 301.5),
        "proc_temp": (308.0, 325.0, 316.5),
        "rpm": (500, 1800, 1150),
        "torque": (30.0, 90.0, 60.0),
        "wear": (20, 230, 100),
        "desc": "Chemical & fluid tank high-viscosity agitator."
    },
    "Rolling Mill Drive Motor": {
        "code": "M-ROLL",
        "air_temp": (300.0, 310.0, 305.0),
        "proc_temp": (315.0, 335.0, 325.0),
        "rpm": (700, 1800, 1250),
        "torque": (80.0, 200.0, 140.0),
        "wear": (50, 300, 160),
        "desc": "Steel & metal rolling mill heavy reduction drive."
    },
    "Precision Grinding Spindle": {
        "code": "M-GRIND",
        "air_temp": (295.0, 302.0, 298.5),
        "proc_temp": (303.0, 312.0, 307.5),
        "rpm": (3000, 7000, 5000),
        "torque": (10.0, 40.0, 25.0),
        "wear": (10, 160, 60),
        "desc": "Ultra-high speed surface & cylindrical grinding spindle."
    },
    "Hydraulic Press Drive": {
        "code": "M-PRESS",
        "air_temp": (298.0, 308.0, 303.0),
        "proc_temp": (310.0, 325.0, 317.5),
        "rpm": (500, 1400, 950),
        "torque": (70.0, 160.0, 115.0),
        "wear": (30, 260, 120),
        "desc": "Heavy metal stamping & forging hydraulic press drive."
    },
    "Packaging Machine Motor": {
        "code": "M-PACK",
        "air_temp": (295.0, 303.0, 299.0),
        "proc_temp": (304.0, 313.0, 308.5),
        "rpm": (1000, 2500, 1750),
        "torque": (15.0, 45.0, 30.0),
        "wear": (10, 180, 70),
        "desc": "High-cycle automated packaging line drive."
    },
    "Textile Spinning Motor": {
        "code": "M-TEXT",
        "air_temp": (296.0, 304.0, 300.0),
        "proc_temp": (305.0, 315.0, 310.0),
        "rpm": (2500, 6000, 4250),
        "torque": (15.0, 50.0, 32.5),
        "wear": (20, 220, 100),
        "desc": "High-velocity yarn spinning & textile frame motor."
    },
    "HVAC Compressor": {
        "code": "M-HVAC",
        "air_temp": (296.0, 305.0, 300.5),
        "proc_temp": (308.0, 325.0, 316.5),
        "rpm": (1200, 3000, 2100),
        "torque": (35.0, 100.0, 67.5),
        "wear": (20, 230, 110),
        "desc": "Commercial building HVAC refrigerant compressor."
    },
    "Industrial Chiller Pump": {
        "code": "M-CHIL",
        "air_temp": (295.0, 303.0, 299.0),
        "proc_temp": (304.0, 315.0, 309.5),
        "rpm": (1000, 2500, 1750),
        "torque": (25.0, 75.0, 50.0),
        "wear": (20, 200, 90),
        "desc": "Cooling plant chilled water circulation pump."
    },
    "Diesel Generator Drive": {
        "code": "M-GEN",
        "air_temp": (298.0, 310.0, 304.0),
        "proc_temp": (320.0, 345.0, 332.5),
        "rpm": (1200, 2200, 1700),
        "torque": (70.0, 180.0, 125.0),
        "wear": (30, 280, 140),
        "desc": "Emergency backup diesel generator engine drive."
    },
    "Wind Turbine Gearbox": {
        "code": "M-WIND",
        "air_temp": (290.0, 305.0, 297.5),
        "proc_temp": (305.0, 325.0, 315.0),
        "rpm": (500, 1800, 1150),
        "torque": (80.0, 200.0, 140.0),
        "wear": (50, 300, 160),
        "desc": "Renewable wind turbine main step-up gearbox."
    },
    "Crane Hoist Motor": {
        "code": "M-HOIST",
        "air_temp": (298.0, 308.0, 303.0),
        "proc_temp": (310.0, 325.0, 317.5),
        "rpm": (500, 1500, 1000),
        "torque": (60.0, 150.0, 105.0),
        "wear": (30, 260, 120),
        "desc": "Heavy overhead gantry crane hoist motor drive."
    },
    "Material Handling Roller Drive": {
        "code": "M-MHD",
        "air_temp": (296.0, 305.0, 300.5),
        "proc_temp": (305.0, 318.0, 311.5),
        "rpm": (800, 2000, 1400),
        "torque": (30.0, 90.0, 60.0),
        "wear": (20, 230, 100),
        "desc": "Warehouse & logistics material sorting roller drive."
    }
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
