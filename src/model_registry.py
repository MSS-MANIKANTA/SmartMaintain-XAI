import joblib
from pathlib import Path
from typing import Any, Tuple, List
from src import config

def save_model(model: Any, filepath: Path = config.BEST_MODEL_PATH):
    """Saves model artifact."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath: Path = config.BEST_MODEL_PATH) -> Any:
    """Loads saved model artifact."""
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found at {filepath}. Please run training first.")
    return joblib.load(filepath)

def save_scaler(scaler: Any, filepath: Path = config.SCALER_PATH):
    """Saves Standard Scaler."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, filepath)
    print(f"Scaler saved to {filepath}")

def load_scaler(filepath: Path = config.SCALER_PATH) -> Any:
    """Loads saved Standard Scaler."""
    if not filepath.exists():
        raise FileNotFoundError(f"Scaler file not found at {filepath}. Please run training first.")
    return joblib.load(filepath)

def save_feature_names(feature_names: List[str], filepath: Path = config.FEATURE_NAMES_PATH):
    """Saves exact list of feature column names in order."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_names, filepath)
    print(f"Feature names saved to {filepath}")

def load_feature_names(filepath: Path = config.FEATURE_NAMES_PATH) -> List[str]:
    """Loads list of feature column names."""
    if not filepath.exists():
        raise FileNotFoundError(f"Feature names file not found at {filepath}. Please run training first.")
    return joblib.load(filepath)
