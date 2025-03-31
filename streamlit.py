import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import time
import gdown

def download_model():
    # Use your extracted Google Drive file ID
gdown.download('https://drive.google.com/uc?id=1T1CCmIQb8ng8qsFWCQRKuLcPw3VysZsE', 'deepfake_cnn_model.h5', quiet=False)


# Call the function to download the model when the app starts
download_model()

from tensorflow.keras.models import load_model

def load_model_from_file():
    model = load_model('deepfake_cnn_model.h5')
    return model
# Custom CSS with dark theme and glow effects
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0f0a1f;
        color: #ffffff;
    }
    
    .title { 
        font-size: 2.8rem;
        background: linear-gradient(45deg, #00ff88, #00ffee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0,255,136,0.3);
    }
    
    .subtitle { 
        font-size: 1.2rem;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .prediction-box {
        background: rgba(15, 10, 31, 0.8);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid #2d3748;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.1);
    }
    
    .prediction-real {
        font-size: 1.8rem;
        color: #00ff88;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-shadow: 0 0 10px rgba(0,255,136,0.3);
    }
    
    .prediction-fake {
        font-size: 1.8rem;
        color: #ff006e;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-shadow: 0 0 10px rgba(255,0,110,0.3);
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #00ff88, #00ffee);
        border: none;
        color: #0f0a1f !important;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(0,255,136,0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 25px rgba(0,255,136,0.5);
    }
    
    .file-uploader {
        border: 2px dashed #2d3748;
        border-radius: 15px;
        padding: 2rem;
        background: rgba(15, 10, 31, 0.5);
    }
    
    .footer {
        text-align: center;
        color: #2d3748;
        padding: 1rem;
        margin-top: 2rem;
    }
    
    .image-container {
        margin: 2rem 0;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0,255,136,0.1);
    }
    
    .clear-button {
        background: linear-gradient(45deg, #ff006e, #ff00ff) !important;
        box-shadow: 0 0 15px rgba(255,0,110,0.3) !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deepfake_cnn_model.h5")

model = load_model()

def predict_image(img):
    # Convert image to RGB if it has an alpha channel (RGBA)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    img = img.resize((256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    label = "✅ Authentic Image" if prediction >= 0.5 else "🚨 Deepfake Detected"
    confidence = prediction * 100 if prediction >= 0.5 else (1 - prediction) * 100
    return label, confidence

# Main Interface
st.markdown('<div class="title">CYBERSCAN</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Distinguish Real Images In Real-Time</div>', unsafe_allow_html=True)

# File Upload
uploaded_file = st.file_uploader(" ", type=["jpg", "png", "jpeg"], 
                               help="Drag & drop media for analysis",
                               key="uploader")

# Action Buttons Grid
col1, col2 = st.columns(2)
with col1:
    analyze_btn = st.button("CHECK", use_container_width=True)
with col2:
    clear_btn = st.button("🔄CLEAR IMAGE", use_container_width=True, 
                        on_click=lambda: st.session_state.clear())

# Display and Processing
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    with st.container():
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze_btn:
        with st.spinner("Decrypting quantum signatures..."):
            time.sleep(1)
            st.session_state['result'] = predict_image(img)

    if 'result' in st.session_state:
        label, confidence = st.session_state['result']
        
        with st.container():
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            
            col_a, col_b = st.columns([3, 2])
            with col_a:
                if "Authentic" in label:
                    st.markdown(f'<div class="prediction-real">⚡ {label}</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-fake">💥 {label}</div>', 
                              unsafe_allow_html=True)
                
                st.metric("CONFIDENCE QUOTIENT", f"{confidence:.2f}%")
                
            with col_b:
                color = "#00ff88" if "Authentic" in label else "#ff006e"
                st.markdown(f"""
                    <div style="height: 10px; border-radius: 5px;
                            background: {color}; width: {confidence}%;
                            box-shadow: 0 0 10px {color}30;
                            transition: width 0.5s ease;">
                    </div>
                """, unsafe_allow_html=True)
                st.caption("NEURAL CERTAINTY INDEX")
            
            st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<div class="footer">CYBERSCAN BY PAVAN N A SHETTY| 03-2025 | DEEP LEARNING</div>', 
          unsafe_allow_html=True)
