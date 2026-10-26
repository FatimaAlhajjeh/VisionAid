# pages/object_detection.py
import streamlit as st
import torch
import torchvision.models as models
from torch import nn
import numpy as np
from PIL import Image
import base64
import cv2

class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", 
               "Dog", "Frog", "Horse", "Ship", "Truck"]

def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(r"C:\Users\majed\capapp\best_model_weights.pth", map_location="cpu"))
    model.eval()
    return model

def object_detection_page(engine):
    model = load_model()
    
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        # Process image
        image = Image.open(uploaded_file).convert("RGB")
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        
        # Prediction
        with torch.no_grad():
            preds = model(tensor)
            pred_idx = torch.argmax(preds).item()
            pred_class = class_names[pred_idx]
        
        # Annotate image
        annotated = np.array(image)
        annotated = cv2.putText(annotated, pred_class, (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Display results
        img_bytes = base64.b64encode(cv2.imencode('.jpg', annotated)[1]).decode()
        
        st.markdown(f"""
            <div class="result-card">
                <div class="result-title">Detection Results</div>
                <div class="image-preview">
                    <img src="data:image/jpeg;base64,{img_bytes}" style="width: 100%;">
                </div>
                <div style="margin-top: 1rem;">
                    <p><strong>Detected Object:</strong></p>
                    <div style="color: var(--accent); font-weight: 500; font-size: 1.25rem;">
                        {pred_class}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        engine.speak(pred_class)