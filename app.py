import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
.stApp {font-family: 'Inter', sans-serif; background: #f5f5f5;}
h1 {text-align: center; color: #333; font-size: 3rem;}
.form-container {background: rgba(255,255,255,0.8); backdrop-filter: blur(20px); border-radius: 20px; padding: 2rem; margin: 2rem 0;}
.stSlider > div > div > div {background: linear-gradient(90deg, #333, #666);}
.stMarkdown p {color: #333;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🏠 **House Price Predictor**")
st.markdown("<p style='text-align:center;color:#666;font-size:1.2rem;'>Interactive SAIT AI Dashboard</p>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load('house_model.pkl')
    le = None
    try:
        le = joblib.load('label_encoder.pkl')
    except:
        pass
    return model, le

model, le = load_model()

# CENTRAL FORM (Main Area)
st.markdown("## 🎛️ **Property Inputs** (Center Panel)")
with st.container():
    col1, col2 = st.columns((1,1))
    
    with col1:
        lot_area = st.slider("🏞️ Lot Area (sqft)", 0, 50000, 10000, help="Land size impact")
        overall_qual = st.slider("⭐ Overall Quality", 1, 10, 5)
        year_built = st.slider("🏗️ Year Built", 1870, 2010, 2000)
    
    with col2:
        first_flr = st.slider("📏 1st Floor SF", 0, 3000, 1200)
        gr_liv = st.slider("🏠 Living Area SF", 0, 5000, 1700)
        garage_cars = st.slider("🚗 Garage Cars", 0, 5, 2)
    
    neighborhood = st.selectbox("🏘️ Neighborhood", ['NAmes', 'CollgCr', 'OldTown', 'Edwards'])
    
    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("🔮 **Predict Price**", use_container_width=True, type="primary"):
        st.session_state.prediction_made = True
    if col_btn2.button("🔄 Reset", use_container_width=True):
        st.rerun()

# Prediction Logic
def predict_price():
    input_df = pd.DataFrame({
        'LotArea': [lot_area], 'OverallQual': [overall_qual], 'YearBuilt': [year_built],
        '1stFlrSF': [first_flr], 'GrLivArea': [gr_liv], 'GarageCars': [garage_cars],
        'Neighborhood': [neighborhood]
    })
    if le:
        try:
            input_df['Neighborhood'] = le.transform(input_df['Neighborhood'])
        except:
            input_df['Neighborhood'] = 0
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0).astype(float)
    return np.exp(model.predict(input_df)[0])

# RESULTS + CHARTS (Below Form)
if 'prediction_made' in st.session_state:
    price = predict_price()
    
    # KPI Row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 **Predicted Price**", f"${price:,.0f}")
    with col2:
        st.metric("📊 Model Accuracy", "RMSE: 0.31")
    with col3:
        st.metric("🔧 Features", len(model.feature_names_in_))
    
    # Charts Row
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        chart_data = pd.DataFrame({
            'Feature': ['Lot Area', 'Quality', 'Living Area'],
            'Value': [lot_area/1000, overall_qual, gr_liv/100]
        })
        fig = px.bar(chart_data, x='Feature', y='Value', title="Key Features")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_chart2:
        fig2 = px.scatter(x=[year_built], y=[price], size=[overall_qual*10], 
                         title="Price vs Build Year", color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig2, use_container_width=True)
    
    # Summary Table
    summary_data = {
        'Feature': ['Lot Area', 'Quality', 'Year Built', '1st Floor', 'Living Area', 'Garage', 'Neighborhood'],
        'Value': [f"{lot_area:,}", overall_qual, year_built, first_flr, gr_liv, garage_cars, neighborhood]
    }
    st.subheader("📋 **Property Summary**")
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

st.markdown("---")

st.markdown("*SAIT Integrated AI | Deployed on Streamlit Cloud*")

