import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="🏠 House Prices Explorer", layout="wide")

st.title("🏠 House Prices Explorer")

# Verify where we are
st.write("**Current Working Directory:**", os.getcwd())

# Load dataset safely
try:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../train.csv"))
    st.success("✅ Dataset loaded successfully!")
    st.write("### Sample Data", df.head())

    # Plot
    st.subheader("💰 Price Distribution")
    fig = px.histogram(df, x="SalePrice", nbins=50, title="Distribution of House Prices")
    st.plotly_chart(fig)

except Exception as e:
    st.error(f"❌ Error loading data: {e}")
