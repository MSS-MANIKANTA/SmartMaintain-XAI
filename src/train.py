import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier

from src import config
from src.data_prep import get_processed_data
from src.evaluate import evaluate_classifier, generate_comparison_report, plot_confusion_matrices
from src import model_registry

def train_and_evaluate_all():
    """Trains classification models, isolation forest anomaly detector, and computes 5-Fold Stratified CV."""
    print("Fetching processed dataset...")
    df = get_processed_data()
    
    # Feature matrix X and target y
    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]
    
    feature_names = list(X.columns)
    print(f"Total samples: {len(df)} | Total features ({len(feature_names)}): {feature_names}")
    
    # Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler and feature names
    model_registry.save_scaler(scaler)
    model_registry.save_feature_names(feature_names)
    
    # 1. Level 2 ML — Isolation Forest Anomaly Detector (Trained on Normal baseline data y=0)
    print("\n-> Training Level 2 ML: Isolation Forest Anomaly Detector...")
    X_normal = X_train_scaled[y_train == 0]
    iso_forest = IsolationForest(
        n_estimators=100, contamination=0.12, random_state=config.RANDOM_STATE
    )
    iso_forest.fit(X_normal)
    model_registry.save_anomaly_detector(iso_forest)
    
    # Calculate pos_weight for XGBoost class imbalance
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / max(num_pos, 1)
    
    # Define Classification Models
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=6, random_state=config.RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=config.RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=scale_pos_weight,
            random_state=config.RANDOM_STATE, eval_metric="logloss"
        )
    }
    
    results = {}
    cv_results_summary = []
    trained_models = {}
    
    # 2. 5-Fold Stratified Cross-Validation Evaluation
    print("\n--- Running 5-Fold Stratified Cross-Validation ---")
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    for name, model in models.items():
        print(f"-> 5-Fold CV for {name}...")
        # Fit on scaled train split
        model.fit(X_train_scaled, y_train)
        
        # Save individual model binary
        sanitized_name = name.lower().replace(" ", "_")
        model_registry.save_model(model, filepath=config.MODELS_DIR / f"{sanitized_name}.joblib")
        
        # Run 5-Fold CV on full scaled dataset
        X_all_scaled = scaler.transform(X)
        cv_scores = cross_validate(model, X_all_scaled, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)
        
        mean_f1 = np.mean(cv_scores["test_f1"])
        std_f1 = np.std(cv_scores["test_f1"])
        mean_auc = np.mean(cv_scores["test_roc_auc"])
        mean_rec = np.mean(cv_scores["test_recall"])
        mean_prec = np.mean(cv_scores["test_precision"])
        mean_acc = np.mean(cv_scores["test_accuracy"])
        
        cv_results_summary.append({
            "Model": name,
            "CV_Mean_F1": round(mean_f1, 4),
            "CV_Std_F1": round(std_f1, 4),
            "CV_Mean_Recall": round(mean_rec, 4),
            "CV_Mean_Precision": round(mean_prec, 4),
            "CV_Mean_Accuracy": round(mean_acc, 4),
            "CV_Mean_ROC_AUC": round(mean_auc, 4)
        })
        
        # Test Set Metrics
        metrics = evaluate_classifier(model, X_test_scaled, y_test)
        results[name] = metrics
        trained_models[name] = model
        print(f"   Test Set Metrics: {metrics}")
        print(f"   5-Fold CV Mean F1: {mean_f1:.4f} ± {std_f1:.4f} | Mean ROC-AUC: {mean_auc:.4f}")
        
    # Export 5-Fold Cross Validation Summary
    df_cv = pd.DataFrame(cv_results_summary).sort_values("CV_Mean_F1", ascending=False)
    config.CROSS_VAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_cv.to_csv(config.CROSS_VAL_PATH, index=False)
    print(f"\nSaved 5-Fold Cross-Validation report to {config.CROSS_VAL_PATH}")
    print(df_cv.to_string(index=False))

    # Generate benchmark table
    df_comparison = generate_comparison_report(results)
    
    # Plot confusion matrices
    plot_confusion_matrices(trained_models, X_test_scaled, y_test)
    
    # Select best model (highest F1-Score)
    best_model_name = df_comparison.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\n[BEST MODEL] Winner: {best_model_name} (F1-Score: {results[best_model_name]['F1-Score']})")
    
    # Save best model to best_model.joblib
    model_registry.save_model(best_model, filepath=config.BEST_MODEL_PATH)
    
    return best_model_name, results

if __name__ == "__main__":
    train_and_evaluate_all()
