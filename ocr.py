# pages/ocr.py
import streamlit as st
import cv2
import numpy as np
from craft_text_detector import Craft
from tensorflow.keras.models import load_model
from openai import OpenAI
import base64
# Update your OCR model loading in ocr.py
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Bidirectional, Attention, Lambda
from tensorflow.keras.utils import custom_object_scope

# Define custom layers
def get_output_shape(input_shape):
    return (input_shape[0], input_shape[2], input_shape[3])

custom_objects = {
    'LSTM': LSTM,
    'Bidirectional': Bidirectional,
    'Attention': Attention,
    'Lambda': Lambda,
    'get_output_shape': get_output_shape
}

# Modify the load_model line
with custom_object_scope(custom_objects):
    model = load_model("ocr_model.hdf5")
client = OpenAI(api_key="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")

craft = Craft(output_dir='./craft_output', crop_type='box', cuda=False)
model = load_model(r"C:\Users\majed\capapp\ocr_model.hdf5")
char_list = sorted(['j','2','c','6','q','1','8','r','v','y','u','m','w','k','4','b','d','a','t',
                    '9','l','0','o','n','h','3','5','i','g','7','f','z','x','p','s','e'])

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 32), interpolation=cv2.INTER_AREA)
    return np.expand_dims(resized / 255.0, axis=(0, -1))

def decode_prediction(pred):
    decoded = K.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0]
    return ''.join([char_list[int(x)] for x in K.get_value(decoded[0]) if int(x) != -1])

def correct_text(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": "Correct this OCR text while maintaining original length. Output only corrected text:"
        }, {
            "role": "user",
            "content": text
        }]
    )
    return response.choices[0].message.content.strip()

def ocr_page(engine):
    uploaded_file = st.file_uploader("Upload Document", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        # Process image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Detect text regions
        prediction_result = craft.detect_text(image)
        text_regions = prediction_result['boxes']
        
        # Process each region
        full_text = []
        for box in text_regions:
            x_min = max(0, int(box[0][0]) - 10)
            y_min = max(0, int(box[0][1]) - 10)
            x_max = min(image.shape[1], int(box[2][0]) + 10)
            y_max = min(image.shape[0], int(box[2][1]) + 10)
            cropped = image[y_min:y_max, x_min:x_max]
            
            processed = preprocess_image(cropped)
            pred = model.predict(processed)
            text = decode_prediction(pred)
            full_text.append(text)
        
        raw_text = " ".join(full_text)
        corrected = correct_text(raw_text)
        
        # Display results
        img_bytes = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()
        
        st.markdown(f"""
            <div class="result-card">
                <div class="result-title">OCR Results</div>
                <div class="image-preview">
                    <img src="data:image/jpeg;base64,{img_bytes}" style="width: 100%;">
                </div>
                <div style="margin-top: 1rem;">
                    <p><strong>Raw Text:</strong> {raw_text}</p>
                    <p style="color: var(--accent); font-weight: 500;">
                        <strong>Corrected:</strong> {corrected}
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        engine.speak(corrected)