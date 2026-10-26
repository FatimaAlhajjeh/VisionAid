import streamlit as st
from pages import ocr , object_detection
from pages.ocr import app as ocr_app
from pages.object_detection import app as object_detection_app

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f7f7f7;
    }
    .phone {
        background: #000;
        width: 360px;
        height: 640px;
        border-radius: 30px;
        margin: auto;
        border: 12px solid #333;
        position: relative;
    }
    .phone-camera {
        background: #f2f2f2;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        position: absolute;
        top: -40px;
        left: 150px;
        border: 6px solid #666;
    }
    .phone-screen {
        background: #fff;
        width: 320px;
        height: 560px;
        border-radius: 10px;
        margin: 20px auto;
        overflow: hidden;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("VisionAid")

# Main phone interface
st.markdown("""
<div class="phone">
    <div class="phone-camera"></div>
    <div class="phone-screen">
        <h3>Welcome to Vision App</h3>
        <p>Select a feature below:</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation using st.selectbox
page = st.selectbox("Choose a feature", ["Main", "OCR", "Object Detection"])

if page == "OCR":
    ocr_app()
elif page == "Object Detection":
    object_detection_app()
