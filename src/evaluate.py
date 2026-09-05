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

def run_live_cross_validation_on_custom_data(df_custom: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Executes live 5-Fold Stratified Cross-Validation on a user-provided dataset.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from src.data_prep import engineer_features

    df_proc = engineer_features(df_custom)
    
    # Identify target column
    target_col = None
    for c in ["target", "Machine failure", "machine_failure", "Target", "Machine_failure"]:
        if c in df_proc.columns:
            target_col = c
            break
            
    if target_col is None or len(df_proc) < 10:
        return None
        
    for col in feature_names:
        if col not in df_proc.columns:
            df_proc[col] = 0.0

    X = df_proc[feature_names].values
    y = df_proc[target_col].values
    
    if len(np.unique(y)) < 2:
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    candidate_models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    }

    n_splits = min(5, int(np.min(np.bincount(y.astype(int)))))
    if n_splits < 2:
        return None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for m_name, clf in candidate_models.items():
        f1_list, recall_list, prec_list, acc_list, auc_list = [], [], [], [], []
        
        for train_idx, val_idx in skf.split(X_scaled, y):
            X_tr, X_va = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_va)
            y_prob = clf.predict_proba(X_va)[:, 1] if hasattr(clf, "predict_proba") else y_pred
            
            acc_list.append(accuracy_score(y_va, y_pred))
            prec_list.append(precision_score(y_va, y_pred, zero_division=0))
            recall_list.append(recall_score(y_va, y_pred, zero_division=0))
            f1_list.append(f1_score(y_va, y_pred, zero_division=0))
            try:
                auc_list.append(roc_auc_score(y_va, y_prob))
            except Exception:
                auc_list.append(0.5)

        results.append({
            "Model": m_name,
            "CV_Mean_F1": round(float(np.mean(f1_list)), 4),
            "CV_Std_F1": round(float(np.std(f1_list)), 4),
            "CV_Mean_Recall": round(float(np.mean(recall_list)), 4),
            "CV_Mean_Precision": round(float(np.mean(prec_list)), 4),
            "CV_Mean_Accuracy": round(float(np.mean(acc_list)), 4),
            "CV_Mean_ROC_AUC": round(float(np.mean(auc_list)), 4)
        })

    return pd.DataFrame(results).sort_values("CV_Mean_F1", ascending=False).reset_index(drop=True)

