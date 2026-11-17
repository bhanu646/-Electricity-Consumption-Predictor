"""
Main Streamlit Application for Electricity Consumption Predictor.
Orchestrates sidebar navigation, session state management, custom CSS styling,
and modular page routing.
"""

import os
import streamlit as st
import pandas as pd

# Page setup - MUST BE FIRST STREAMLIT CALL
st.set_page_config(
    page_title="Electricity Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Relative directory resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "electricity_consumption.csv")
ROOT_DATA_PATH = os.path.join(BASE_DIR, "electricity_consumption.csv")
CSS_PATH = os.path.join(BASE_DIR, "assets", "style.css")

# Import modular pages
from pages.dashboard import render_dashboard
from pages.dataset import render_dataset_page
from pages.visualization import render_visualization_page
from pages.model_training import render_model_training_page
from pages.prediction import render_prediction_page
from pages.history import render_history_page
from pages.about import render_about_page

from src.data_preprocessing import load_data
from src.train_model import train_and_evaluate_all, load_saved_model
from src.database import init_db

# Load custom CSS
def load_css():
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH, 'r') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize SQLite database
init_db()

# State Management: Load default dataset into session state if not already loaded
if 'active_df' not in st.session_state:
    try:
        if os.path.exists(DATA_PATH):
            st.session_state['active_df'] = load_data(DATA_PATH)
        elif os.path.exists(ROOT_DATA_PATH):
            st.session_state['active_df'] = load_data(ROOT_DATA_PATH)
        else:
            # Generate dataset dynamically if missing
            from data.generate_data import generate_electricity_dataset
            df_gen = generate_electricity_dataset()
            st.session_state['active_df'] = df_gen
            os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
            df_gen.to_csv(DATA_PATH, index=False)
    except Exception as e:
        st.error(f"Error initializing dataset: {str(e)}")
        st.session_state['active_df'] = None

# State Management: Auto-train or load models on startup
if 'trained_state' not in st.session_state:
    st.session_state['trained_state'] = None
    if st.session_state['active_df'] is not None:
        try:
            # Check if saved model exists on disk first
            saved_pipeline, metadata = load_saved_model()
            if saved_pipeline is not None and metadata is not None:
                # Pre-train or evaluate all 3 to populate comparison table
                st.session_state['trained_state'] = train_and_evaluate_all(st.session_state['active_df'])
            else:
                # Initial training
                st.session_state['trained_state'] = train_and_evaluate_all(st.session_state['active_df'])
        except Exception as e:
            # Graceful error handling
            st.session_state['trained_state'] = None

# Sidebar Navigation
st.sidebar.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
        <h2 style="color: #2563eb; margin: 0; font-size: 1.6rem;">⚡ PowerPredict</h2>
        <span style="font-size: 0.8rem; color: #64748b; font-weight: 500;">B.Tech Machine Learning Project</span>
    </div>
""", unsafe_allow_html=True)

nav_options = [
    "🏠 Home / Dashboard",
    "📂 Dataset Explorer",
    "📊 Data Visualization",
    "🤖 Model Training",
    "⚡ Predict Consumption",
    "📜 Prediction History",
    "ℹ️ About Project"
]

selected_page = st.sidebar.radio(
    "Navigate to:",
    options=nav_options,
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
    **Project Info:**
    * **Algorithm:** Multi-Model Regression
    * **Target:** Electricity Consumption (kWh)
    * **Framework:** Scikit-Learn + Streamlit
""")

st.sidebar.markdown("""
    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2rem;">
        Developed for Academic Demonstration & Viva Voce
    </div>
""", unsafe_allow_html=True)

# Main router
df = st.session_state['active_df']
trained_state = st.session_state['trained_state']

if df is None:
    st.error("No active dataset loaded. Please check that `data/electricity_consumption.csv` is accessible.")
else:
    if selected_page == "🏠 Home / Dashboard":
        render_dashboard(df, trained_state)
    elif selected_page == "📂 Dataset Explorer":
        render_dataset_page(df)
    elif selected_page == "📊 Data Visualization":
        render_visualization_page(df)
    elif selected_page == "🤖 Model Training":
        render_model_training_page(df)
    elif selected_page == "⚡ Predict Consumption":
        render_prediction_page(trained_state)
    elif selected_page == "📜 Prediction History":
        render_history_page()
    elif selected_page == "ℹ️ About Project":
        render_about_page()
