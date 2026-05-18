
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="HR Attrition Dashboard", page_icon="👥", layout="wide")

@st.cache_data
def load_and_train():
    df = pd.read_csv("IBM HR Analytics.csv")
    df_copy = df.copy()
    df_copy.drop(columns=['EmployeeCount', 'Over18', 'StandardHours'], inplace=True)
    df_copy['Attrition'] = df_copy['Attrition'].map({'Yes': 1, 'No': 0})
    df_copy['PromotionRisk'] = (df_copy['YearsSinceLastPromotion'] >= 3).astype(int)
    df_copy['HikeRisk'] = (df_copy['PercentSalaryHike'] <= 14).astype(int)
    df_copy['OvertimeRisk'] = ((df_copy['OverTime'] == 'Yes') & (df_copy['MonthlyIncome'] < 5000)).astype(int)
    df_eda = df_copy.copy()
    le = LabelEncoder()
    for col in ['OverTime', 'Gender', 'BusinessTravel']:
        df_copy[col] = le.fit_transform(df_copy[col])
    df_copy = pd.get_dummies(df_copy, columns=['Department', 'JobRole', 'MaritalStatus', 'EducationField'], drop_first=True)
    for col in ['TextFeature', 'AgeGroup', 'EducationLevel', 'SatisfactionLevel']:
        if col in df_copy.columns:
            df_copy.drop(columns=[col], inplace=True)
    X = df_copy.drop(columns=['Attrition'])
    y = df_copy['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    model = XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred) * 100, 1),
        "F1 Score": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_pred), 4),
    }
    risk_scores = model.predict_proba(X_test)[:, 1]
    df_risk = X_test.copy()
    df_risk['AttritionRisk'] = (risk_scores * 100).round(2)
    df_risk['ActualAttrition'] = y_test.values
    df_risk['RiskLevel'] = pd.cut(df_risk['AttritionRisk'], bins=[0, 30, 60, 100], labels=['Low', 'Medium', 'High'])
    return df_eda, model, X.columns.tolist(), metrics, df_risk

df_eda, model, feature_cols, metrics, df_risk = load_and_train()

st.sidebar.title("👥 HR Attrition App")
st.sidebar.markdown("Built by **Rishika Sinha**")
page = st.sidebar.radio("Navigate", ["📊 Overview", "🔍 Predict Employee Risk", "📈 Feature Importance"])

if page == "📊 Overview":
    st.title("📊 HR Employee Attrition Dashboard")
    st.markdown("**Dataset:** IBM HR Analytics | **Model:** XGBoost + SMOTE")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", "1,470")
    col2.metric("Attrition Rate", "16.1%")
    col3.metric("Model Accuracy", f"{metrics['Accuracy']}%")
    col4.metric("ROC-AUC", metrics['ROC-AUC'])
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Attrition Rate by Department")
        dept = df_eda.groupby('Department')['Attrition'].mean().reset_index()
        dept['Attrition'] *= 100
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=dept, x='Department', y='Attrition', palette='Reds_r', ax=ax)
        ax.set_ylabel("Attrition Rate (%)")
        ax.set_xlabel("")
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=9)
        st.pyplot(fig)
    with col_b:
        st.subheader("Attrition Rate by Marital Status")
        ms = df_eda.groupby('MaritalStatus')['Attrition'].mean().reset_index()
        ms['Attrition'] *= 100
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=ms, x='MaritalStatus', y='Attrition', palette='Greens_r', ax=ax)
        ax.set_ylabel("Attrition Rate (%)")
        ax.set_xlabel("")
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=9)
        st.pyplot(fig)
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Attrition Rate by Overtime")
        ot = df_eda.groupby('OverTime')['Attrition'].mean().reset_index()
        ot['Attrition'] *= 100
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=ot, x='OverTime', y='Attrition', palette='Purples_r', ax=ax)
        ax.set_ylabel("Attrition Rate (%)")
        ax.set_xlabel("")
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=9)
        st.pyplot(fig)
    with col_d:
        st.subheader("Risk Level Distribution")
        risk_order = ['Low', 'Medium', 'High']
        rc = df_risk['RiskLevel'].value_counts().reindex(risk_order).reset_index()
        rc.columns = ['RiskLevel', 'Count']
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'}
        sns.barplot(data=rc, x='RiskLevel', y='Count', palette=[colors[r] for r in rc['RiskLevel']], ax=ax)
        for i, row in rc.iterrows():
            ax.text(i, row['Count'] + 1, str(row['Count']), ha='center', fontweight='bold')
        ax.set_xlabel("")
        st.pyplot(fig)
    st.divider()
    st.subheader("🔤 TF-IDF: Keywords for Employees Who Left vs Stayed")
    df_tfidf = df_eda.copy()
    df_tfidf['TextFeature'] = df_tfidf['JobRole'] + ' ' + df_tfidf['Department'] + ' ' + df_tfidf['EducationField']
    left_text = ' '.join(df_tfidf[df_tfidf['Attrition'] == 1]['TextFeature'].values)
    stayed_text = ' '.join(df_tfidf[df_tfidf['Attrition'] == 0]['TextFeature'].values)
    tfidf = TfidfVectorizer(stop_words='english', max_features=10)
    tfidf_matrix = tfidf.fit_transform([left_text, stayed_text])
    features = tfidf.get_feature_names_out()
    left_scores = tfidf_matrix[0].toarray()[0]
    stayed_scores = tfidf_matrix[1].toarray()[0]
    tfidf_df = pd.DataFrame({'Feature': features, 'Left_Score': left_scores, 'Stayed_Score': stayed_scores}).sort_values('Left_Score', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(tfidf_df))
    width = 0.35
    ax.bar([i - width/2 for i in x], tfidf_df['Left_Score'], width=width, color='#e74c3c', label='Left')
    ax.bar([i + width/2 for i in x], tfidf_df['Stayed_Score'], width=width, color='#2ecc71', label='Stayed')
    ax.set_xticks(x)
    ax.set_xticklabels(tfidf_df['Feature'], rotation=45)
    ax.set_title('TF-IDF Keywords: Employees Who Left vs Stayed')
    ax.set_xlabel('Keywords')
    ax.set_ylabel('TF-IDF Score')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

elif page == "🔍 Predict Employee Risk":
    st.title("🔍 Predict Employee Attrition Risk")
    st.markdown("Fill in the employee details below to get a risk prediction.")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 60, 35)
        monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000, step=500)
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        years_since_promo = st.slider("Years Since Last Promotion", 0, 15, 2)
    with col2:
        overtime = st.selectbox("OverTime", ["Yes", "No"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    with col3:
        job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
        education_field = st.selectbox("Education Field", ["Life Sciences", "Other", "Medical", "Marketing", "Technical Degree", "Human Resources"])
        percent_hike = st.slider("Percent Salary Hike", 11, 25, 15)
        work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4], index=2)
        distance_home = st.slider("Distance from Home", 1, 30, 5)
    if st.button("🔮 Predict Risk", use_container_width=True):
        input_data = {col: 0 for col in feature_cols}
        numeric_map = {
            'Age': age, 'MonthlyIncome': monthly_income,
            'JobSatisfaction': job_satisfaction, 'YearsAtCompany': years_at_company,
            'YearsSinceLastPromotion': years_since_promo, 'PercentSalaryHike': percent_hike,
            'WorkLifeBalance': work_life_balance, 'DistanceFromHome': distance_home,
            'OverTime': 1 if overtime == "Yes" else 0,
            'Gender': 1 if gender == "Male" else 0,
            'BusinessTravel': {'Non-Travel': 0, 'Travel_Frequently': 1, 'Travel_Rarely': 2}[business_travel],
        }
        for k, v in numeric_map.items():
            if k in input_data:
                input_data[k] = v
        input_data['PromotionRisk'] = 1 if years_since_promo >= 3 else 0
        input_data['HikeRisk'] = 1 if percent_hike <= 14 else 0
        input_data['OvertimeRisk'] = 1 if (overtime == "Yes" and monthly_income < 5000) else 0
        for col_name, val in [(f"Department_{department}", 1), (f"JobRole_{job_role}", 1), (f"MaritalStatus_{marital_status}", 1), (f"EducationField_{education_field}", 1)]:
            if col_name in input_data:
                input_data[col_name] = val
        input_df = pd.DataFrame([input_data])[feature_cols]
        prob = model.predict_proba(input_df)[0][1] * 100
        if prob >= 60:
            risk_level, color, emoji = "HIGH RISK", "#e74c3c", "🔴"
        elif prob >= 30:
            risk_level, color, emoji = "MEDIUM RISK", "#f39c12", "🟡"
        else:
            risk_level, color, emoji = "LOW RISK", "#2ecc71", "🟢"
        st.divider()
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.markdown(f"""
            <div style='background-color:{color}22; border-left: 6px solid {color};
                        padding: 20px; border-radius: 8px; text-align:center'>
                <h1 style='color:{color}'>{emoji} {risk_level}</h1>
                <h2 style='color:{color}'>{prob:.1f}% Attrition Probability</h2>
            </div>
            """, unsafe_allow_html=True)
        with col_res2:
            st.markdown("### Risk Factors Detected")
            if input_data['PromotionRisk']:
                st.warning(f"⚠️ No promotion in {years_since_promo} years")
            if input_data['HikeRisk']:
                st.warning(f"⚠️ Low salary hike ({percent_hike}%)")
            if input_data['OvertimeRisk']:
                st.warning(f"⚠️ Overtime with low income")
            if marital_status == "Single":
                st.info("ℹ️ Single employees have higher attrition rates")
            if department == "Sales":
                st.info("ℹ️ Sales department has highest attrition rate")
            if not any([input_data['PromotionRisk'], input_data['HikeRisk'], input_data['OvertimeRisk']]):
                st.success("✅ No major risk flags detected")

elif page == "📈 Feature Importance":
    st.title("📈 Top 15 Features Driving Attrition")
    st.markdown("Based on XGBoost + SMOTE model — higher score = more influence on prediction.")
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=importance, x='Importance', y='Feature', palette='rocket', ax=ax)
    ax.set_xlabel("Feature Importance Score")
    ax.set_ylabel("")
    ax.set_title("Top 15 Features — XGBoost + SMOTE", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    st.divider()
    st.subheader("Model Performance Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{metrics['Accuracy']}%")
    m2.metric("F1 Score", metrics['F1 Score'])
    m3.metric("ROC-AUC", metrics['ROC-AUC'])
    st.caption("Model: XGBoost + SMOTE | Dataset: IBM HR Analytics | Built by Rishika Sinha")