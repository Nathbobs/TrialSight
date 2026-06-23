"""
TrialSight Streamlit app entry point.
Run with: streamlit run app/main.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so relative imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="TrialSight",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar branding
with st.sidebar:
    st.markdown("## 🔬 TrialSight")
    st.markdown(
        "Clinical trial data intelligence platform powered by "
        "ClinicalTrials.gov, Qdrant, and Groq Llama 3.3."
    )
    st.divider()
    st.markdown(
        "**Architecture:**\n"
        "- Bronze → Silver → Gold ETL\n"
        "- Supabase PostgreSQL\n"
        "- Qdrant vector search\n"
        "- Groq Llama 3.3 70B RAG\n"
    )
    st.divider()
    st.caption("Data from ClinicalTrials.gov · Not medical advice")

# Tab layout
tab_dashboard, tab_chat = st.tabs(["📊 Dashboard", "💬 RAG Chat"])

with tab_dashboard:
    from app.dashboard import render_dashboard
    render_dashboard()

with tab_chat:
    from app.chat import render_chat
    render_chat()
