import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_features.csv"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def run_eda():
    print(f"Loading processed dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nTarget Class Distribution:")
    print(df["target"].value_counts(normalize=True))
    
    # 1. Class Balance Plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x="target", hue="target", data=df, palette=["#00CC96", "#FF4B4B"], legend=False)
    plt.title("Industrial Sensor Dataset Target Class Balance (0: Normal, 1: Failure)", fontweight="bold")
    plt.xlabel("Machine Status (0: Normal, 1: Failure)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_balance.png", dpi=300)
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Sensor Feature Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()
    
    # 3. Key Feature Distributions by Target
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    features_to_plot = ["temp_difference_k", "power_kw", "torque_nm", "tool_wear_min"]
    
    for idx, feat in enumerate(features_to_plot):
        ax = axes[idx // 2, idx % 2]
        sns.boxplot(x="target", y=feat, hue="target", data=df, ax=ax, palette=["#00CC96", "#FF4B4B"], legend=False)
        ax.set_title(f"{feat} by Target Class", fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_boxplots.png", dpi=300)
    plt.close()
    
    print(f"\nAll EDA figures saved cleanly to {FIGURES_DIR}")

if __name__ == "__main__":
    run_eda()
