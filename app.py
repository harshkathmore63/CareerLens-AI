import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import re
import ast
import warnings
warnings.filterwarnings('ignore')

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="CareerLens AI", layout="wide", page_icon="🚀")

# ================== LOAD MODELS ==================
@st.cache_resource
def load_models():
    with open('salary_predictor_enhanced.pkl', 'rb') as f:
        salary = pickle.load(f)
    with open('job_title_classifier_final.pkl', 'rb') as f:
        clf = pickle.load(f)
    return salary, clf

salary_artifacts, clf_artifacts = load_models()

# Salary model
salary_model = salary_artifacts['model']
salary_scaler = salary_artifacts['scaler']
skill_vectorizer = salary_artifacts['vectorizer']
skill_svd = salary_artifacts['svd']

# Classifier model
clf_model = clf_artifacts['model']
clf_scaler = clf_artifacts['scaler']
le = clf_artifacts['label_encoder']

title_vectorizer = clf_artifacts['title_vectorizer']
title_svd = clf_artifacts['title_svd']
skill_vectorizer_clf = clf_artifacts['skill_vectorizer']
skill_svd_clf = clf_artifacts['skill_svd']

top_roles = clf_artifacts.get('top_roles', [])

# ================== LOAD DATA ==================
@st.cache_data
def load_data():
    df = pd.read_csv("careerlens_mega_enriched.csv")

    df['parsed_skills'] = df['parsed_skills'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

    def extract_exp_years(text):
        if pd.isna(text):
            return np.nan
        nums = re.findall(r'\d+\.?\d*', str(text))
        nums = [float(n) for n in nums if 0 <= float(n) <= 40]
        return np.mean(nums) if nums else np.nan

    df['exp_years'] = df['experience_std'].apply(extract_exp_years)

    TECH_SKILLS = {'PYTHON','JAVA','SQL','AWS','DOCKER','LINUX','MYSQL','CSS','HTML','GIT','EXCEL','REACT','NODE','DJANGO','SPRING','TABLEAU','POWERBI','KUBERNETES','AZURE','GCP','DEVOPS','SPARK','HADOOP','TENSORFLOW','C++','C#'}

    df['tech_skills'] = df['parsed_skills'].apply(
        lambda lst: [s.strip().upper().replace(' ', '') for s in lst if s.strip().upper().replace(' ', '') in TECH_SKILLS]
    )

    df = df[df['tech_skills'].map(len) > 0].copy()

    return df

df = load_data()

# ================== SIDEBAR ==================
with st.sidebar:
    st.title("🚀 CareerLens AI")

    page = st.radio("Navigate", [
        "Dashboard",
        "Salary Predictor",
        "Job Match"
    ])

# ================== DASHBOARD ==================
if page == "Dashboard":

    st.title("CareerLens AI Dashboard")
    st.markdown("""
    ### 🚀 About CareerLens AI

    CareerLens AI is an intelligent career guidance platform that:
    - Predicts **job roles** based on your skills
    - Estimates **salary potential**
    - Analyzes **market trends from 50K+ job listings**

    Built using Machine Learning, NLP, and real-world job data.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Jobs", f"{len(df):,}")
    col2.metric("Avg Salary", f"₹{df['salary_inr'].mean():,.0f}")
    col3.metric("Top Skill", df['tech_skills'].explode().value_counts().index[0])

    st.subheader("Salary Distribution")
    fig = px.histogram(df, x='salary_inr', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

# ================== JOB MATCH ==================
if page == "Job Match":

    st.title("🎯 Find Your Best Career Match")

    st.markdown("Enter your skills and optionally a job title to discover the most suitable roles.")

    col1, col2 = st.columns(2)

    with col1:
        user_title = st.text_input("💼 Job Title (optional)")

    with col2:
        user_skills = st.text_area("🛠 Skills (comma separated)")

    if st.button("🔍 Analyze My Profile"):

        if not user_skills.strip():
            st.warning("Please enter your skills")
        else:
            try:
                skills_list = [s.strip() for s in user_skills.split(",") if s.strip()]

                title_vec = title_vectorizer.transform([user_title])
                title_red = title_svd.transform(title_vec)

                skill_vec = skill_vectorizer_clf.transform([" ".join(skills_list)])
                skill_red = skill_svd_clf.transform(skill_vec)

                X = np.hstack([title_red, skill_red])
                X_scaled = clf_scaler.transform(X)

                probs = clf_model.predict_proba(X_scaled)[0]
                top_indices = np.argsort(probs)[-5:][::-1]
                top_roles_pred = le.inverse_transform(top_indices)

                st.markdown("## 🚀 Top Career Matches")

                for i, role in enumerate(top_roles_pred):
                    confidence = probs[top_indices[i]] * 100
                    st.progress(int(confidence))
                    st.write(f"**{role}** — {confidence:.2f}% match")

                if top_roles:
                    st.markdown("### 🔥 Trending Roles")
                    st.info(", ".join(top_roles[:10]))

            except Exception as e:
                st.error(f"Error: {e}")
# ================== SALARY PREDICTOR ==================
if page == "Salary Predictor":

    st.title("💰 Estimate Your Salary")

    st.markdown("Provide your skills and experience to get an estimated salary range.")

    col1, col2 = st.columns(2)

    with col1:
        user_skills = st.text_area("🛠 Skills")

    with col2:
        user_exp = st.slider("📈 Experience (years)", 0, 20, 2)

    if st.button("💸 Predict Salary"):

        if not user_skills.strip():
            st.warning("Please enter skills")
        else:
            try:
                skills_list = [s.strip().upper() for s in user_skills.split(",") if s.strip()]

                skill_vec = skill_vectorizer.transform([" ".join(skills_list)])
                skill_red = skill_svd.transform(skill_vec)

                exp_array = np.array([[user_exp]])
                X = np.hstack([skill_red, exp_array])

                X_scaled = salary_scaler.transform(X)
                pred_salary = salary_model.predict(X_scaled)[0]

                st.markdown("## 💡 Estimated Salary")

                st.success(f"₹ {int(pred_salary):,} per year")

                # bonus insight
                if pred_salary < 500000:
                    st.info("📊 Entry-level range")
                elif pred_salary < 1200000:
                    st.info("📊 Mid-level range")
                else:
                    st.info("📊 High-paying role 🚀")

            except Exception as e:
                st.error(f"Error: {e}")
# ================== FOOTER ==================
st.markdown("---")
st.markdown("CareerLens AI | ML Powered Career Insights")