# HR Attrition Prediction App

## Overview
An interactive web application that predicts employee attrition risk using 
Machine Learning, built with Streamlit. HR teams can visualize attrition 
patterns and predict individual employee risk in real time.

## Live Demo
🚀 https://rishika-hr-attrition.streamlit.app

## Features
- 📊 **Overview Dashboard** — KPI metrics, attrition charts by department, 
  marital status, overtime, and risk distribution
- 🔤 **TF-IDF Analysis** — NLP keyword comparison between employees who 
  left vs stayed
- 🔍 **Employee Risk Predictor** — Input employee details and get 
  High/Medium/Low risk prediction instantly
- 📈 **Feature Importance** — Top 15 features driving attrition with 
  model performance summary

## Tech Stack
- Python
- Streamlit
- XGBoost
- SMOTE (Imbalanced-learn)
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- TF-IDF (Scikit-learn)

## Model Details
- **Algorithm:** XGBoost + SMOTE
- **Accuracy:** 85.7%
- **ROC-AUC:** 0.6393
- **Class Imbalance:** Handled using SMOTE
- **Risk Buckets:** High (>60%), Medium (30-60%), Low (<30%)

## Custom Engineered Features
| Feature | Logic | Business Reason |
|---|---|---|
| PromotionRisk | YearsSinceLastPromotion >= 3 | No growth = higher exit risk |
| HikeRisk | PercentSalaryHike <= 14% | Low hike band had highest attrition |
| OvertimeRisk | Overtime = Yes AND Income < 5000 | Compounding stress factor |

## How to Run Locally

**1. Clone the repository**

    git clone https://github.com/rishika-sinha1315/hr-attrition-app.git

**2. Install dependencies**

    pip install -r requirements.txt

**3. Run the app**

    streamlit run app.py
