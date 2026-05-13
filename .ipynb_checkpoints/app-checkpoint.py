import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import re
import warnings
warnings.filterwarnings('ignore')

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="CareerLens AI", layout="wide", page_icon="rocket")

# ================== LOAD MODELS ==================
@st.cache_resource
def load_models():
    with open('salary_predictor_enhanced.pkl', 'rb') as f:
        salary = pickle.load(f)
    with open('job_title_classifier_final.pkl', 'rb') as f:
        clf = pickle.load(f)
    return salary, clf


salary_artifacts, clf_artifacts = load_models()
salary_model = salary_artifacts['model']
salary_scaler = salary_artifacts['scaler']
skill_vectorizer = salary_artifacts['vectorizer']
skill_svd = salary_artifacts['svd']

clf_model = clf_artifacts['model']
clf_scaler = clf_artifacts['scaler']
le = clf_artifacts['label_encoder']

# ================== LOAD & PROCESS DATA ==================
@st.cache_data
def load_data():
    # ✅ FIXED PATH (IMPORTANT)
    df = pd.read_csv("careerlens_mega_enriched.csv")

    df['parsed_skills'] = df['parsed_skills'].apply(lambda x: eval(x) if isinstance(x, str) else [])

    # === EXTRACT exp_years ===
    def extract_exp_years(text):
        if pd.isna(text):
            return np.nan
        nums = re.findall(r'\d+\.?\d*', str(text))
        nums = [float(n) for n in nums if 0 <= float(n) <= 40]
        return np.mean(nums) if nums else np.nan

    df['exp_years'] = df['experience_std'].apply(extract_exp_years)

    level_to_exp = {'entry': 1, 'mid': 3.5, 'senior': 7, 'expert': 12}

    for level in df['experience_level'].unique():
        mask = (df['experience_level'] == level) & (df['exp_years'].isna())
        df.loc[mask, 'exp_years'] = level_to_exp.get(level, 5)

    # === TECH SKILLS ===
    TECH_SKILLS = {'PYTHON','JAVA','SQL','AWS','DOCKER','LINUX','MYSQL','CSS','HTML','GIT','EXCEL','REACT','NODE','DJANGO','SPRING','TABLEAU','POWERBI','KUBERNETES','AZURE','GCP','DEVOPS','SPARK','HADOOP','TENSORFLOW','C++','C#'}

    def extract_tech(lst):
        return [s.strip().upper().replace(' ', '') for s in lst if s.strip().upper().replace(' ', '') in TECH_SKILLS]

    df['tech_skills'] = df['parsed_skills'].apply(extract_tech)
    df = df[df['tech_skills'].map(len) > 0].copy()

    # === ENCODINGS ===
    df['company_enc'] = df.groupby('company_std')['salary_inr'].transform('mean')
    df['location_enc'] = df.groupby('location_std')['salary_inr'].transform('mean')

    return df


df = load_data()

# ================== SIDEBAR ==================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/rocket.png")
    st.title("CareerLens AI")

    page = st.radio("Navigate", [
        "Dashboard",
        "Salary Predictor",
        "Job Match",
        "Skill Forecaster",
        "Career Recommender",
        "Analytics Hub"
    ])

# ================== DASHBOARD ==================
if page == "Dashboard":

    st.title("CareerLens AI Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Jobs", f"{len(df):,}")

    with col2:
        st.metric("Avg Salary", f"₹{df['salary_inr'].mean():,.0f}")

    with col3:
        st.metric("Top Skill", df['tech_skills'].explode().value_counts().index[0])

    with col4:
        st.metric("Fastest Growing", "LINUX +241%")

    dash_tab1, dash_tab2 = st.tabs(["Salary & Skills", "Job Postings by Location"])

    with dash_tab1:

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Salary Distribution")
            fig = px.histogram(df, x='salary_inr', nbins=50,
                               title="Salary Distribution (INR)",
                               color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Top 10 Skills Demand")
            skill_count = pd.Series([s for sublist in df['tech_skills'] for s in sublist]).value_counts().head(10)

            fig = px.bar(
                x=skill_count.values,
                y=skill_count.index,
                orientation='h',
                title="Most In-Demand Skills",
                color_discrete_sequence=['#FF6B6B']
            )
            st.plotly_chart(fig, use_container_width=True)

    with dash_tab2:

        st.markdown("### Job Postings by Location")
        st.markdown("**Where are the most opportunities?**")

        salary_range = st.slider(
            "Filter by Salary Range (₹LPA)",
            min_value=0,
            max_value=int(df['salary_inr'].max() / 100000),
            value=(5, 50),
            step=5,
            key="dash_loc_slider"
        )

        min_sal, max_sal = [x * 100000 for x in salary_range]

        loc_df = df[(df['salary_inr'] >= min_sal) & (df['salary_inr'] <= max_sal)]
        loc_counts = loc_df['location_std'].value_counts().head(20)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Top 20 Cities")

            fig_bar = px.bar(
                x=loc_counts.index,
                y=loc_counts.values,
                labels={'x': 'City', 'y': 'Job Postings'},
                title="Top Job Locations",
                color=loc_counts.values,
                color_continuous_scale="Viridis"
            )

            fig_bar.update_layout(height=500)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.markdown("#### Job Density Map (India)")

            city_coords = {
                'BENGALURU': (12.97, 77.59),
                'MUMBAI': (19.07, 72.87),
                'DELHI': (28.70, 77.10),
                'HYDERABAD': (17.38, 78.48),
                'CHENNAI': (13.08, 80.27),
                'PUNE': (18.52, 73.85),
                'GURGAON': (28.45, 77.02),
                'NOIDA': (28.53, 77.39),
                'KOLKATA': (22.57, 88.36),
                'AHMEDABAD': (23.02, 72.57),
                'JAIPUR': (26.91, 75.78),
                'CHANDIGARH': (30.73, 76.77)
            }

            map_data = []

            for city, count in loc_counts.items():
                if city.upper() in city_coords:
                    lat, lon = city_coords[city.upper()]
                    map_data.append({
                        'City': city,
                        'Count': count,
                        'lat': lat,
                        'lon': lon
                    })

            map_df = pd.DataFrame(map_data)

            if not map_df.empty:

                fig_map = px.scatter_mapbox(
                    map_df,
                    lat="lat",
                    lon="lon",
                    size="Count",
                    color="Count",
                    hover_name="City",
                    size_max=40,
                    zoom=4,
                    title="Job Postings Density",
                    color_continuous_scale="Plasma",
                    mapbox_style="carto-positron"
                )

                fig_map.update_layout(height=500,
                                      margin={"r":0,"t":40,"l":0,"b":0})

                st.plotly_chart(fig_map, use_container_width=True)

            else:
                st.info("No data in selected salary range")

# ================== FOOTER ==================
st.markdown("---")
st.markdown("**CareerLens AI** — Built with Love | Powered by xAI | Data: 54K+ Jobs")