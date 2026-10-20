import os
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Fraud Command Center",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Seamless Fullscreen CSS: Removes all Streamlit chrome, paddings, gutters, and scrollbar clipping
st.markdown("""
<style>
    /* Hide Streamlit Header, Toolbar, Footer, Sidebar, and decorations */
    header,
    [data-testid="stHeader"],
    [data-testid="stSidebar"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    /* Remove all margins, paddings, and scrollbars on the host Streamlit window */
    html, body, .stApp {
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
        background-color: #fafafa !important;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    .main,
    .block-container,
    div[data-testid="stVerticalBlock"],
    div[data-testid="stCustomComponentV1"] {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100vw !important;
        width: 100vw !important;
        height: 100vh !important;
        overflow: hidden !important;
    }

    /* Position the iframe fixed across the entire viewport */
    iframe {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
        z-index: 999999 !important;
    }
</style>
""", unsafe_allow_html=True)

# Path to Fraud Command Center.html
html_path = os.path.join(os.path.dirname(__file__), "..", "Fraud Command Center.html")
if not os.path.exists(html_path):
    html_path = "Fraud Command Center.html"

with open(html_path, "r", encoding="utf-8") as f:
    html_code = f.read()

# Render Fraud Command Center HTML
components.html(html_code, scrolling=True)
