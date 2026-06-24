"""
TrialSight Streamlit app entry point.
Run with: streamlit run app/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from streamlit_float import float_init

st.set_page_config(
    page_title="TrialSight",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

float_init()

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

# Main content — Dashboard
from app.dashboard import render_dashboard
render_dashboard()

# Initialise chat open/close state
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

# Floating chat panel (rendered before FAB so FAB appears on top)
if st.session_state.chat_open:
    from app.chat import render_chat_panel
    panel = st.container()
    with panel:
        render_chat_panel()
    panel.float(
        "position: fixed; bottom: 6rem; right: 2rem; z-index: 9998; "
        "width: 420px; max-height: calc(100vh - 9rem); overflow-y: auto; "
        "background-color: #262730; "
        "border-radius: 12px; "
        "box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4); "
        "padding: 1rem 1.25rem;"
    )

# Force FAB button into a circular shape
st.markdown("""
<style>
[data-testid="stBaseButton-primary"] {
    border-radius: 50% !important;
    width: 3.5rem !important;
    height: 3.5rem !important;
    min-height: 3.5rem !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stBaseButton-primary"] p {
    font-size: 1.4rem !important;
    margin: 0 !important;
    line-height: 1 !important;
}
</style>
""", unsafe_allow_html=True)

# Floating FAB button — 💬 to open, ✕ to close
fab = st.container()
with fab:
    label = "✕" if st.session_state.chat_open else "💬"
    if st.button(label, key="chat_fab", type="primary"):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()
fab.float(
    "position: fixed; bottom: 2rem; right: 2rem; z-index: 9999; width: 3.5rem;"
)
