import streamlit as stlt
from tabs import dashboards, predict

stlt.set_page_config(page_title="MLOps Rakuten Project", layout="wide")
stlt.title("Project Overview")

tab1, tab2 = stlt.tabs(["📊 Dashboards", "🔮 Model & API"])

with tab1:
    dashboards.render()
with tab2:
    predict.render()