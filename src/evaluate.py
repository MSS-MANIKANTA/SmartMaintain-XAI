import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from src import config

def evaluate_classifier(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Calculates evaluation metrics for a single trained model."""
    y_pred = model.predict(X_test)
    
    # Get failure probabilities for positive class (1)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc_auc = 0.5
        
    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4)
    }

def generate_comparison_report(results: Dict[str, Dict[str, float]], save_path: Path = config.MODEL_COMPARISON_PATH) -> pd.DataFrame:
    """Generates comparison dataframe and exports to CSV."""
    df_res = pd.DataFrame.from_dict(results, orient="index")
    df_res.index.name = "Model"
    df_res = df_res.reset_index()
    
    # Sort by F1-Score descending
    df_res = df_res.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(save_path, index=False)
    print("\n--- Model Benchmark Results ---")
    print(df_res.to_string(index=False))
    print(f"\nSaved comparison report to {save_path}")
    return df_res

def plot_confusion_matrices(models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, save_dir: Path = config.FIGURES_DIR):
    """Plots and saves confusion matrix figures for all models."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False,
                    xticklabels=["Normal", "Failure"], yticklabels=["Normal", "Failure"])
        axes[idx].set_title(f"{name} Confusion Matrix", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")
        
    plt.tight_layout()
    cm_path = save_dir / "confusion_matrices.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrices figure to {cm_path}")
