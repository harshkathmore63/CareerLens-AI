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

    st.title("🎯 Job Role Predictor")

    user_title = st.text_input("Enter Job Title (optional)")
    user_skills = st.text_area("Enter Skills (comma separated)")

    if st.button("Predict Job Role"):

        if not user_skills.strip():
            st.warning("Please enter skills")
        else:
            try:
                skills_list = [s.strip() for s in user_skills.split(",") if s.strip()]

                title_vec = title_vectorizer.transform([user_title])
                title_red = title_svd.transform(title_vec)

                skill_vec = skill_vectorizer_clf.transform([" ".join(skills_list)])
                skill_red = skill_svd_clf.transform(skill_vec)

                X = np.hstack([title_red, skill_red])

                # ✅ padding fix
                expected_features = clf_model.n_features_in_

                if X.shape[1] < expected_features:
                    padding = np.zeros((1, expected_features - X.shape[1]))
                    X = np.hstack([X, padding])
                elif X.shape[1] > expected_features:
                    X = X[:, :expected_features]

                skills_text = " ".join(skills_list).lower()

                if any(word in skills_text for word in ["machine learning", "deep learning", "nlp", "ai"]):
                    roles = ["Data Scientist", "ML Engineer", "AI Engineer"]
                elif any(word in skills_text for word in ["python", "sql", "excel", "tableau"]):
                    roles = ["Data Analyst", "Data Scientist", "Business Analyst"]
                elif any(word in skills_text for word in ["html", "css", "javascript", "react", "node"]):
                    roles = ["Frontend Developer", "Full Stack Developer", "Web Developer"]
                elif any(word in skills_text for word in ["aws", "docker", "kubernetes", "devops"]):
                    roles = ["DevOps Engineer", "Cloud Engineer", "Site Reliability Engineer"]
                elif any(word in skills_text for word in ["java", "spring", "backend"]):
                    roles = ["Backend Developer", "Java Developer", "Software Engineer"]
                else:
                    probs = clf_model.predict_proba(X)[0]
                    top_indices = np.argsort(probs)[-5:][::-1]
                    roles = le.inverse_transform(top_indices)

                st.success("Top Career Matches:")

                filtered_roles = []
                for role in roles:
                    if role.lower() not in ["not specified", "others", "unknown"]:
                        filtered_roles.append(role)

                for i, role in enumerate(filtered_roles[:5]):
                    st.write(f"{i+1}. {role}")

            except Exception as e:
                st.error(f"Error: {e}")
# ================== SALARY PREDICTOR ==================
if page == "Salary Predictor":

    st.title("💰 Salary Predictor")

    user_skills = st.text_area("Enter Skills (comma separated)")
    user_exp = st.slider("Years of Experience", 0, 20, 2)

    if st.button("Predict Salary"):

        if not user_skills.strip():
            st.warning("Please enter skills")
        else:
            try:
                skills_list = [s.strip().upper() for s in user_skills.split(",") if s.strip()]

                skill_vec = skill_vectorizer.transform([" ".join(skills_list)])
                skill_red = skill_svd.transform(skill_vec)

                exp_array = np.array([[user_exp]])

                X = np.hstack([skill_red, exp_array])

                # ✅ padding fix
                expected_features = salary_model.n_features_in_

                if X.shape[1] < expected_features:
                    padding = np.zeros((1, expected_features - X.shape[1]))
                    X = np.hstack([X, padding])
                elif X.shape[1] > expected_features:
                    X = X[:, :expected_features]

                pred_salary = salary_model.predict(X)[0]
                exp_multiplier = 1 + (user_exp * 0.08)
                adjusted_salary = pred_salary * exp_multiplier

                num_skills = len(skills_list)
                if num_skills <= 2:
                    adjusted_salary *= 0.5
                elif num_skills <= 4:
                    adjusted_salary *= 0.7
                elif num_skills <= 6:
                    adjusted_salary *= 0.9
                else:
                    adjusted_salary *= 1.1

                high_value_skills = ["python", "machine learning", "aws", "data science"]
                bonus = sum(1 for s in skills_list if s.lower() in high_value_skills)
                adjusted_salary *= (1 + bonus * 0.05)


                if user_exp <= 1:
                    adjusted_salary = min(adjusted_salary, 500000)
                elif user_exp <= 3:
                    adjusted_salary = min(adjusted_salary, 900000)
                adjusted_salary = max(adjusted_salary, 250000)
                st.success(f"Estimated Salary: ₹{int(adjusted_salary):,}")

                if adjusted_salary < 500000:
                    st.info("📊 Entry Level Salary")
                elif adjusted_salary < 1200000:
                    st.info("📊 Mid Level Salary")
                else:
                    st.info("📊 High Paying Role 🚀")

            except Exception as e:
                st.error(f"Error: {e}")
# ================== FOOTER ==================
st.markdown("---")
st.markdown("CareerLens AI | ML Powered Career Insights")