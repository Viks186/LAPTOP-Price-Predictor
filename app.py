import streamlit as st
import pickle                   #python -m streamlit run app.py
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(page_title="Laptop Predictor", page_icon="💻", layout="wide")

# Load the model and dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Advanced CSS for Background and Glassmorphism
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), 
                    url("https://images.unsplash.com/photo-1603302576837-37561b2e2302?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTh8fGxhcHRvcHxlbnwwfHwwfHx8MA%3D%3D");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 25px;
        margin-bottom: 20px;
    }

    .stSelectbox, .stSlider, .stNumberInput {
        color: white !important;
    }

    h1, h2, h3, p {
        color: white !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .stButton > button {
        width: 100%;
        border-radius: 50px;
        height: 3.5em;
        background: linear-gradient(45deg, #00c6ff, #0072ff);
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(0, 198, 255, 0.4);
    }

    .prediction-box {
        background: rgba(0, 255, 127, 0.2);
        border: 2px solid #00ff7f;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# App Header
st.title("💻 Laptop Price Predictor")
st.markdown("##### Fill in the specifications to Get the market value")
st.write("---")

# Form Inputs
col1, col2 = st.columns(2)

with col1:
    st.markdown('## ⚙️ Core Identity')
    company = st.selectbox('Select Brand', df['Company'].unique())
    type_name = st.selectbox('Type of Laptop', df['TypeName'].unique())
    os = st.selectbox('Operating System', df['os'].unique())
    cpu = st.selectbox('CPU Processor', df['Cpu brand'].unique())
    gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())

with col2:
    st.markdown('### 🛠️ Hardware & Display')
    ram = st.select_slider('RAM (GB)', options=[2, 4, 6, 8, 12, 16, 24, 32, 64], value=8)
    
    inner_col1, inner_col2 = st.columns(2)
    with inner_col1:
        hdd = st.selectbox('HDD (GB)', [0, 128, 256, 512, 1024, 2048])
        touchscreen = st.checkbox('Touchscreen')
    with inner_col2:
        ssd = st.selectbox('SSD (GB)', [0, 8, 128, 256, 512, 1024])
        ips = st.checkbox('IPS Panel')
    
    weight = st.number_input('Weight of Laptop (kg)', min_value=0.5, max_value=5.0, value=1.5)
    
    res_col1, res_col2 = st.columns([2, 1])
    with res_col1:
        resolution = st.selectbox('Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
    with res_col2:
        screen_size = st.number_input('Screen Size (inches)', value=15.6)

st.write("---")

# Prediction Action
if st.button('🚀 Predict Market Price'):
    ts = 1 if touchscreen else 0
    ips_val = 1 if ips else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    query = pd.DataFrame([[
        company, type_name, ram, weight, ts, ips_val, ppi, cpu, hdd, ssd, gpu, os
    ]], columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'])

    prediction = np.exp(pipe.predict(query)[0])
    
    st.balloons()
    st.markdown(f"""
        <div class="prediction-box">
            <h2 style="color: #00ff7f !important; margin: 0;">Estimated Price</h2>
            <h1 style="color: white !important; font-size: 50px;">₹ {int(prediction):,}</h1>
        </div>
    """, unsafe_allow_html=True)
