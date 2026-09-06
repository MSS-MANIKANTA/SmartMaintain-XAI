import math
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from src import config

def download_raw_data(url: str = config.UCI_DATASET_URL, output_path: Path = config.RAW_DATA_PATH) -> Path:
    """Downloads the authentic UCI AI4I 2020 Predictive Maintenance dataset if not present or empty."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Raw UCI dataset already exists at {output_path} ({output_path.stat().st_size} bytes)")
        return output_path

    print(f"Downloading authentic UCI AI4I 2020 dataset from {url}...")
    try:
        from ucimlrepo import fetch_ucirepo
        ai4i = fetch_ucirepo(id=601)
        df_features = ai4i.data.features
        df_targets = ai4i.data.targets
        df_all = pd.concat([df_features, df_targets], axis=1)
        df_all.to_csv(output_path, index=False)
        print(f"Dataset successfully fetched via ucimlrepo and saved to {output_path}")
        return output_path
    except Exception as e1:
        print(f"ucimlrepo fetch failed/not installed ({e1}), falling back to direct URL download...")

    try:
        import requests
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Dataset successfully saved to {output_path} ({output_path.stat().st_size} bytes)")
    except Exception as e2:
        print(f"requests download failed ({e2}), trying urllib...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(output_path, 'wb') as f:
            f.write(resp.read())
        print(f"Dataset successfully saved via urllib to {output_path} ({output_path.stat().st_size} bytes)")
    
    return output_path

def load_raw_data(filepath: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Loads raw dataset and cleans column names."""
    if not filepath.exists():
        download_raw_data(output_path=filepath)
    
    df = pd.read_csv(filepath)
    df = df.rename(columns=config.COLUMN_MAPPING)
    if "Machine_Type" not in df.columns:
        df["Machine_Type"] = "CNC Milling Spindle (High Precision)"
    if "Machine_ID" not in df.columns:
        df["Machine_ID"] = [f"M-UCI-{i+1:05d}" for i in range(len(df))]
    print(f"Loaded raw dataset shape: {df.shape}")
    return df

def generate_multi_machine_dataset(samples_per_machine: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Generates 12,500 synthetic records across 25 machine types with realistic normal & 5 failure patterns."""
    np.random.seed(random_state)
    records = []
    
    for m_name, specs in config.MACHINE_25_SPECS.items():
        code = specs["code"]
        a_min, a_max, _ = specs["air_temp"]
        p_min, p_max, _ = specs["proc_temp"]
        r_min, r_max, _ = specs["rpm"]
        t_min, t_max, _ = specs["torque"]
        w_min, w_max, _ = specs["wear"]
        
        n_normal = int(samples_per_machine * 0.8)  # 400 normal records (80%)
        n_fail = samples_per_machine - n_normal   # 100 failure records (20%)
        n_per_pattern = n_fail // 5              # 20 records per failure pattern
        
        # 1. NORMAL Records (Label 0)
        for i in range(n_normal):
            air_t = np.random.uniform(a_min, a_max)
            proc_t = np.random.uniform(p_min, p_max)
            rpm = np.random.uniform(r_min, r_max)
            torque = np.random.uniform(t_min, t_max)
            wear = np.random.uniform(w_min, w_max)
            quality = np.random.choice(["H", "M", "L"], p=[0.2, 0.5, 0.3])
            
            records.append({
                "Machine_ID": f"{code}-{i+1:03d}",
                "Machine_Type": m_name,
                "air_temperature_k": round(air_t, 2),
                "process_temperature_k": round(proc_t, 2),
                "rotational_speed_rpm": round(rpm, 1),
                "torque_nm": round(torque, 2),
                "tool_wear_min": int(wear),
                "type": quality,
                "target": 0
            })
            
        # 2. FAILURE Records (Label 1) - Pattern 1: Temperature-related Failure
        for i in range(n_per_pattern):
            air_t = np.random.uniform(a_max + 2.0, a_max + 10.0)
            proc_t = np.random.uniform(p_max + 5.0, p_max + 18.0)
            rpm = np.random.uniform(r_min, r_max)
            torque = np.random.uniform(t_min, t_max)
            wear = np.random.uniform(w_min, w_max * 0.8)
            quality = np.random.choice(["H", "M", "L"], p=[0.2, 0.5, 0.3])
            
            records.append({
                "Machine_ID": f"{code}-F1-{i+1:03d}",
                "Machine_Type": m_name,
                "air_temperature_k": round(air_t, 2),
                "process_temperature_k": round(proc_t, 2),
                "rotational_speed_rpm": round(rpm, 1),
                "torque_nm": round(torque, 2),
                "tool_wear_min": int(wear),
                "type": quality,
                "target": 1
            })
            
        # Pattern 2: Mechanical Overload Failure (High Torque & Temp)
        for i in range(n_per_pattern):
            air_t = np.random.uniform(a_min + 1.0, a_max + 4.0)
            proc_t = np.random.uniform(p_min + 4.0, p_max + 10.0)
            rpm = np.random.uniform(r_min, r_max * 0.9)
            torque = np.random.uniform(t_max * 1.25, t_max * 1.9)
            wear = np.random.uniform(w_min, w_max * 0.8)
            quality = np.random.choice(["H", "M", "L"], p=[0.2, 0.5, 0.3])
            
            records.append({
                "Machine_ID": f"{code}-F2-{i+1:03d}",
                "Machine_Type": m_name,
                "air_temperature_k": round(air_t, 2),
                "process_temperature_k": round(proc_t, 2),
                "rotational_speed_rpm": round(rpm, 1),
                "torque_nm": round(torque, 2),
                "tool_wear_min": int(wear),
                "type": quality,
                "target": 1
            })
            
        # Pattern 3: High-Speed Over-RPM Failure (High RPM & Torque)
        for i in range(n_per_pattern):
            air_t = np.random.uniform(a_min, a_max)
            proc_t = np.random.uniform(p_min, p_max)
            rpm = np.random.uniform(r_max * 1.2, r_max * 1.7)
            torque = np.random.uniform(t_max * 1.1, t_max * 1.5)
            wear = np.random.uniform(w_min, w_max * 0.8)
            quality = np.random.choice(["H", "M", "L"], p=[0.2, 0.5, 0.3])
            
            records.append({
                "Machine_ID": f"{code}-F3-{i+1:03d}",
                "Machine_Type": m_name,
                "air_temperature_k": round(air_t, 2),
                "process_temperature_k": round(proc_t, 2),
                "rotational_speed_rpm": round(rpm, 1),
                "torque_nm": round(torque, 2),
                "tool_wear_min": int(wear),
                "type": quality,
                "target": 1
            })
            
        # Pattern 4: Tool-Wear Failure (Excessive Tool Wear)
        for i in range(n_per_pattern):
            air_t = np.random.uniform(a_min, a_max)
            proc_t = np.random.uniform(p_min, p_max)
            rpm = np.random.uniform(r_min, r_max)
            torque = np.random.uniform(t_min * 1.1, t_max * 1.2)
            wear = np.random.uniform(w_max + 15, w_max + 90)
            quality = np.random.choice(["H", "M", "L"], p=[0.2, 0.5, 0.3])
            
            records.append({
                "Machine_ID": f"{code}-F4-{i+1:03d}",
                "Machine_Type": m_name,
                "air_temperature_k": round(air_t, 2),
                "process_temperature_k": round(proc_t, 2),
                "rotational_speed_rpm": round(rpm, 1),
                "torque_nm": round(torque, 2),
                "tool_wear_min": int(wear),
                "type": quality,
                "target": 1
            })

        # Pattern 5: Combined Multi-Factor Failure
        for i in range(n_per_pattern):
            air_t = np.random.uniform(a_max + 1.0, a_max + 6.0)
            proc_t = np.random.uniform(p_max + 4.0, p_max + 12.0)
            rpm = np.random.uniform(r_max * 1.1, r_max * 1.4)
            torque = np.random.uniform(t_max * 1.15, t_max * 1.6)
            wear = np.random.uniform(w_max + 10, w_max + 60)
            quality = np.random.choice(["H", "M", "L"], p=[0.2, 0.5, 0.3])
            
            records.append({
                "Machine_ID": f"{code}-F5-{i+1:03d}",
                "Machine_Type": m_name,
                "air_temperature_k": round(air_t, 2),
                "process_temperature_k": round(proc_t, 2),
                "rotational_speed_rpm": round(rpm, 1),
                "torque_nm": round(torque, 2),
                "tool_wear_min": int(wear),
                "type": quality,
                "target": 1
            })

    df = pd.DataFrame(records)
    synth_path = config.RAW_DATA_DIR / "multi_machine_25.csv"
    df.to_csv(synth_path, index=False)
    print(f"Generated 25-Machine dataset with {len(df)} records -> {synth_path}")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers physics and machine-relative stress features for predictive maintenance with robust column mapping & deduplication."""
    data = df.copy()
    
    # 0. Deduplicate initial column labels
    data = data.loc[:, ~data.columns.duplicated()].copy()
    
    # 1. Normalize column names cleanly using fuzzy & exact mapping
    norm_cols = {}
    for col in data.columns:
        c_clean = str(col).strip()
        c_lower = c_clean.lower()
        
        if c_clean in config.COLUMN_MAPPING:
            norm_cols[col] = config.COLUMN_MAPPING[c_clean]
        elif ("process" in c_lower or "proc" in c_lower) and "temp" in c_lower:
            norm_cols[col] = "process_temperature_k"
        elif "air" in c_lower and "temp" in c_lower:
            norm_cols[col] = "air_temperature_k"
        elif "speed" in c_lower or "rpm" in c_lower or "rotation" in c_lower:
            norm_cols[col] = "rotational_speed_rpm"
        elif "torque" in c_lower or "nm" in c_lower:
            norm_cols[col] = "torque_nm"
        elif "wear" in c_lower or "tool" in c_lower:
            norm_cols[col] = "tool_wear_min"
        elif c_lower in ["type", "quality", "variant", "product_type"]:
            norm_cols[col] = "type"
        elif c_lower in ["machine failure", "failure", "target", "target_col", "machine_failure"]:
            norm_cols[col] = "target"

    if norm_cols:
        data = data.rename(columns=norm_cols)
        # Deduplicate again in case multiple original columns mapped to the same target name
        data = data.loc[:, ~data.columns.duplicated()].copy()

    # Defaults for missing essential features
    defaults = {
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500,
        "torque_nm": 40.0,
        "tool_wear_min": 0,
        "type": "M"
    }
    for col_name, def_val in defaults.items():
        if col_name not in data.columns:
            data[col_name] = def_val
        data[col_name] = data[col_name].fillna(def_val)

    # Convert numeric columns safely
    num_cols = ["air_temperature_k", "process_temperature_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]
    for nc in num_cols:
        data[nc] = pd.to_numeric(data[nc], errors="coerce").fillna(defaults[nc])

    # 1. Temperature difference (Heat dissipation indicator)
    data["temp_difference_k"] = data["process_temperature_k"] - data["air_temperature_k"]
    
    # 2. Mechanical Power output (kW): Power = (2 * pi * N * T) / 60000
    data["power_kw"] = (2 * math.pi * data["rotational_speed_rpm"] * data["torque_nm"]) / 60000.0
    
    # 3. Tool wear rate relative to rotational speed
    data["wear_rate"] = data["tool_wear_min"] / (data["rotational_speed_rpm"] + 1e-5)
    
    # 4. Temperature to Torque ratio
    data["temp_torque_ratio"] = data["process_temperature_k"] / (data["torque_nm"] + 1e-5)

    # 5. Machine-Relative Stress Ratios (Normalizing against machine specs)
    rpm_max_vals = []
    torque_max_vals = []
    wear_max_vals = []
    temp_max_vals = []

    for idx, row in data.iterrows():
        m_type = row.get("Machine_Type", row.get("machine_type", None))
        if pd.notna(m_type) and str(m_type) in config.MACHINE_25_SPECS:
            specs = config.MACHINE_25_SPECS[str(m_type)]
            r_max = specs["rpm"][1]
            t_max = specs["torque"][1]
            w_max = specs["wear"][1]
            p_max = specs["proc_temp"][1]
        else:
            r_max, t_max, w_max, p_max = 3000.0, 65.0, 180.0, 313.0
            
        rpm_max_vals.append(r_max)
        torque_max_vals.append(t_max)
        wear_max_vals.append(w_max)
        temp_max_vals.append(p_max)

    data["rpm_stress"] = data["rotational_speed_rpm"] / (np.array(rpm_max_vals) + 1e-5)
    data["torque_stress"] = data["torque_nm"] / (np.array(torque_max_vals) + 1e-5)
    data["wear_stress"] = data["tool_wear_min"] / (np.array(wear_max_vals) + 1e-5)
    data["temp_stress"] = data["process_temperature_k"] / (np.array(temp_max_vals) + 1e-5)
    
    # Safe One-hot encoding for machine quality 'type' (L, M, H)
    for t_col in ["type_H", "type_L", "type_M"]:
        if t_col not in data.columns:
            data[t_col] = 0

    if "type" in data.columns:
        type_dummies = pd.get_dummies(data["type"].astype(str), prefix="type", dtype=int)
        for t_col in ["type_H", "type_L", "type_M"]:
            if t_col in type_dummies.columns:
                data[t_col] = type_dummies[t_col]
        data = data.drop(columns=["type"])

    # Final deduplication check to prevent any downstream pandas reindexing errors
    data = data.loc[:, ~data.columns.duplicated()].copy()
    
    return data

def get_processed_data(force_reprocess: bool = False) -> pd.DataFrame:
    """Orchestrates data downloading, synthetic generation, feature engineering, and saving to disk."""
    if config.PROCESSED_DATA_PATH.exists() and not force_reprocess:
        print(f"Loading processed dataset from {config.PROCESSED_DATA_PATH}")
        return pd.read_csv(config.PROCESSED_DATA_PATH)
    
    # 1. Load authentic UCI raw dataset
    df_uci = load_raw_data()
    
    # 2. Generate 12,500 multi-machine synthetic dataset across 25 equipment models
    df_multi = generate_multi_machine_dataset(samples_per_machine=500)
    
    # 3. Combine raw datasets
    df_combined = pd.concat([df_uci, df_multi], ignore_index=True)
    print(f"Combined UCI ({len(df_uci)}) + Multi-Machine ({len(df_multi)}) = {len(df_combined)} total raw records.")
    
    # 4. Feature Engineering
    df_engineered = engineer_features(df_combined)
    
    # Drop non-feature metadata and UCI specific failure target sub-columns
    drop_cols = [
        "udi", "product_id", "twf", "hdf", "pwf", "osf", "rnf", 
        "TWF", "HDF", "PWF", "OSF", "RNF", "Machine_ID", "Machine_Type", 
        "UDI", "Product ID"
    ]
    existing_drop_cols = [col for col in df_engineered.columns if col in drop_cols]
    df_processed = df_engineered.drop(columns=existing_drop_cols)
    df_processed = df_processed.dropna()
    
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print(f"Processed dataset saved to {config.PROCESSED_DATA_PATH} with shape {df_processed.shape}")
    
    return df_processed

if __name__ == "__main__":
    download_raw_data()
    df_proc = get_processed_data(force_reprocess=True)
    print("Class breakdown in target:")
    print(df_proc[config.TARGET_COL].value_counts(normalize=True))
