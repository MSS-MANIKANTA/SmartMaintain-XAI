# SmartMaintain-XAI ⚙️
### Explainable AI-Based Predictive Maintenance System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![XAI: SHAP](https://img.shields.io/badge/XAI-SHAP-green.svg)](https://shap.readthedocs.io/)
[![Docker: Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

---

## 📌 Executive Summary

**SmartMaintain-XAI** is an enterprise-grade industrial predictive maintenance decision-support platform. It ingests telemetry data from industrial machine sensors (air temperature, process temperature, rotational speed, torque, tool wear, quality grades across 25 machine models), predicts the probability of equipment failure, categorizes machine health into a 4-tier risk system (**LOW**, **MEDIUM**, **HIGH**, **CRITICAL**), uses **SHAP (Shapley Additive exPlanations)** to identify the specific sensor parameters driving each risk assessment, and provides targeted maintenance protocols.

---

## ⚙️ Key Platform Features

1. **25 Equipment Models Catalog**: Pre-calibrated normal operating parameters for 25 industrial machine models (spindles, hydraulic pumps, compressors, turbines, extruders, motors, gearboxes).
2. **Machine-Relative Stress Feature Engineering**: Normalizes raw sensor readings against each equipment model's design limits (`rpm_stress`, `torque_stress`, `wear_stress`, `temp_stress`).
3. **Random Forest Machine Learning Engine**: High-performance classifier trained on 22,500 industrial records.
4. **SHAP Explainable AI (XAI)**: Quantifies the exact marginal impact of each sensor reading on the failure probability.
5. **Actionable Maintenance Recommendations**: Automatically maps top failure drivers to specific mechanical, thermal, electrical, and tool replacement protocols.
6. **Batch CSV Screening & Reporting**: Upload bulk CSV datasets, screen multiple machines simultaneously, filter by risk level, and export diagnostic CSV reports.

---

## 📊 Verified Model Benchmark Results

Evaluation conducted on a 22,500-sample industrial dataset (authentic baseline combined with multi-machine synthetic telemetry):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Production Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🏆 **Random Forest** | **99.24%** | **96.19%** | **97.89%** | **0.9703** | **0.9993** | **Production Winner** |
| ⚡ **XGBoost Classifier** | 99.04% | 94.72% | 97.89% | 0.9628 | 0.9992 | Benchmark Champion |
| 🌲 **Decision Tree** | 97.47% | 85.03% | 97.01% | 0.9062 | 0.9957 | Tree Baseline |
| 📈 **Logistic Regression** | 91.67% | 61.64% | 89.96% | 0.7316 | 0.9712 | Linear Baseline |

*Source of truth: `reports/model_comparison.csv`*

---

## 🏗️ Architecture & Pipeline Workflow

```
               [ Industrial Telemetry & Sensor Ingestion ]
                                    │
                                    ▼
                [ Preprocessing & Stress Ratio Engineering ]
      (Power kW | Temp Diff | Wear Rate | Machine-Relative Stress Ratios)
                                    │
                                    ▼
                   [ Production Random Forest Model ]
                                    │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
      [ Failure Probability ]                  [ SHAP Explainer ]
               │                                (Shapley Driver Values)
               └────────────────────┬────────────────────┘
                                    ▼
                 [ 4-Tier Risk Matrix Categorization ]
                 (🟢 LOW | 🟡 MEDIUM | 🟠 HIGH | 🔴 CRITICAL)
                                    │
                                    ▼
                  [ Actionable Maintenance Protocol ]
                                    │
                                    ▼
                [ Streamlit Production Dashboard UI ]
```

---

## 📁 Repository Structure

```
SmartMaintain-XAI/
├── app/
│   └── dashboard.py          # Production Streamlit UI
├── src/
│   ├── __init__.py
│   ├── config.py             # Path settings & 25-machine specifications
│   ├── data_prep.py          # Ingestion & feature engineering
│   ├── train.py              # Training & benchmark pipeline
│   ├── evaluate.py           # Evaluation metrics & confusion matrix generator
│   ├── explain.py            # SHAP explainer & visualization
│   ├── risk.py               # 4-Tier risk matrix & recommendations
│   └── model_registry.py     # Joblib serializer/deserializer
├── models/
│   ├── best_model.joblib     # Production Random Forest model
│   ├── scaler.joblib         # StandardScaler object
│   └── feature_names.joblib # Feature column list
├── reports/
│   ├── figures/              # Benchmark confusion matrix plots
│   └── model_comparison.csv  # Verified metrics table (Single Source of Truth)
├── data/
│   ├── raw/                  # Raw telemetry datasets
│   └── processed/            # Processed feature dataset
├── notebooks/                # Development & EDA scripts
│   └── 01_eda.py
├── .env.example              # Production environment variables template
├── .gitignore                # Version control ignore rules
├── Dockerfile                # Docker container configuration
├── requirements.txt          # Production Python dependencies
└── README.md                 # Product documentation
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-org/SmartMaintain-XAI.git
cd SmartMaintain-XAI
pip install -r requirements.txt
```

### 2. Run Data Pipeline & Train Models
Generate datasets and train production models:
```bash
# Ingest data & engineer features
python -m src.data_prep

# Train & benchmark models
python -m src.train
```

### 3. Launch Local Dashboard
Start the production Streamlit web application:
```bash
streamlit run app/dashboard.py
```
App will be accessible at: `http://localhost:8501`

---

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t smartmaintain-xai .
```

### Run Container
```bash
docker run -d -p 8501:8501 --name smartmaintain-app smartmaintain-xai
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for details.
