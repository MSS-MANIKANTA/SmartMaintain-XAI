import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src import config
from src.data_prep import get_processed_data
from src.evaluate import evaluate_classifier, generate_comparison_report, plot_confusion_matrices
from src import model_registry

def train_and_evaluate_all():
    """Trains 4 models, evaluates performance, saves benchmarks and best model."""
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
    
    # Calculate pos_weight for XGBoost class imbalance
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / max(num_pos, 1)
    
    # Define models
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
    trained_models = {}
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        print(f"-> Training {name}...")
        # Linear models use scaled features, Tree models work fine with unscaled or scaled
        # We pass scaled features for consistency
        model.fit(X_train_scaled, y_train)
        
        # Save individual model binary
        sanitized_name = name.lower().replace(" ", "_")
        model_registry.save_model(model, filepath=config.MODELS_DIR / f"{sanitized_name}.joblib")
        
        # Evaluate
        metrics = evaluate_classifier(model, X_test_scaled, y_test)
        results[name] = metrics
        trained_models[name] = model
        print(f"   Metrics: {metrics}")
        
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
