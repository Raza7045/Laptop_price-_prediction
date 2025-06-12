import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the pre-trained model and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Set Streamlit page configuration
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# Title and Introduction
st.title("Laptop Price Predictor 💻")
st.markdown("### Predict the price of laptops based on your configuration.")

# Input Fields (User Specifications)
st.markdown("### Select Laptop Specifications")

# Brand
company = st.selectbox('💼 Brand', df['Company'].unique())

# Type of Laptop
laptop_type = st.selectbox('📌 Type', df['TypeName'].unique())

# RAM Size (in GB)
ram = st.selectbox('🛠 RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Laptop Weight (kg)
weight = st.number_input('⚖ Weight (kg)', min_value=0.5, max_value=5.0, step=0.1)

# Touchscreen
touchscreen = st.selectbox('🖥 Touchscreen', ['No', 'Yes'])

# IPS Display
ips = st.selectbox('🔳 IPS Display', ['No', 'Yes'])

# Screen Size (inches)
screen_size = st.slider('📏 Screen Size (inches)', 10.0, 18.0, 13.0)

# Screen Resolution
resolution = st.selectbox('🖥 Screen Resolution', 
                           ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# Display PPI Calculation
X_res, Y_res = map(int, resolution.split('x'))
ppi = np.sqrt(X_res**2 + Y_res**2) / screen_size
st.markdown(f"### 💡 Pixels per inch (PPI): {ppi:.2f} PPI")

# CPU
cpu = st.selectbox('💻 CPU', df['Cpu brand'].unique())

# HDD Storage (in GB)
hdd = st.selectbox('💾 HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD Storage (in GB)
ssd = st.selectbox('🔋 SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU Brand
gpu = st.selectbox('🎮 GPU', df['Gpu brand'].unique())

# Operating System
os = st.selectbox('🖥 Operating System', df['os'].unique())

# Real-time Summary of Input Configuration
st.markdown("### Your Laptop Configuration Summary")
st.write(f"**Brand**: {company}")
st.write(f"**Type**: {laptop_type}")
st.write(f"**RAM**: {ram} GB")
st.write(f"**Weight**: {weight} kg")
st.write(f"**Touchscreen**: {'Yes' if touchscreen == 'Yes' else 'No'}")
st.write(f"**IPS Display**: {'Yes' if ips == 'Yes' else 'No'}")
st.write(f"**Screen Size**: {screen_size} inches")
st.write(f"**Resolution**: {resolution}")
st.write(f"**CPU**: {cpu}")
st.write(f"**HDD**: {hdd} GB")
st.write(f"**SSD**: {ssd} GB")
st.write(f"**GPU**: {gpu}")
st.write(f"**OS**: {os}")

# Predict Button
if st.button('💰 Predict Price'):
    try:
        # Process Inputs
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0

        # Prepare query for prediction
        query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
        query = query.reshape(1, -1)

        # Prediction and Output
        predicted_price = np.exp(pipe.predict(query)[0])

        # Display predicted price
        st.markdown(f"### 💰 The predicted price of this configuration is: ₹{int(predicted_price):,}")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Add a progress bar to make the experience more engaging
with st.spinner('We are calculating the price... Please wait'):
    # Simulating a delay for the prediction process
    import time
    time.sleep(2)  # Simulate the prediction delay for better interactivity
