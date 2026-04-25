import streamlit as st
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
from model import FERModel  # Importing your provided model class

# --- Page Config ---
st.set_page_config(
    page_title="Emotion AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for UI Polish ---
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    .main-header {
        font-size: 2.5rem;
        color: #4B4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .emotion-text {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model (Cached for performance) ---
@st.cache_resource
def load_fer_model():
    model = FERModel()
    # Load weights with map_location to ensure it works on CPU even if trained on GPU
    model.load_state_dict(torch.load("fer_model.pth", map_location="cpu"))
    model.eval()
    return model

try:
    model = load_fer_model()
except FileNotFoundError:
    st.error("Error: 'fer_model.pth' or 'model.py' not found. Please make sure they are in the same directory.")
    st.stop()

# --- Constants & Setup ---
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_COLORS = {
    'Angry': '#FF4B4B',    # Red
    'Disgust': '#2E8B57',  # SeaGreen
    'Fear': '#800080',     # Purple
    'Happy': '#FFA500',    # Orange
    'Neutral': '#808080',  # Grey
    'Sad': '#4169E1',      # RoyalBlue
    'Surprise': '#FFD700'  # Gold
}

# Pre-processing (Copied from your webcam_demo.py to match training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Main App Layout ---
st.markdown('<div class="main-header">Real-time Facial Expression Recognition</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📷 Live Camera Feed")
    # Placeholder for the video feed
    st_frame = st.image([]) 
    
    # Control buttons
    start_btn = st.button('Start Camera', type="primary")
    stop_btn = st.button('Stop Camera')

with col2:
    st.markdown("### 📊 Emotion Analytics")
    current_emotion_text = st.empty()
    chart_placeholder = st.empty()
    
    # Initialize a default chart
    df_init = pd.DataFrame({"Probability": [0.0]*7, "Emotion": EMOTIONS})
    chart_placeholder.bar_chart(df_init.set_index("Emotion"))

# --- Logic Loop ---
if start_btn and not stop_btn:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not access the webcam.")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Frame Read Error")
                break
            
            # Flip frame for mirror effect and convert to RGB
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(
                gray_frame, 
                scaleFactor=1.3, 
                minNeighbors=5, 
                minSize=(30, 30)
            )

            current_probs = np.zeros(7)
            dominant_emotion = "Waiting..."
            
            # Process faces
            for (x, y, w, h) in faces:
                # 1. Draw Rectangle
                cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 2. Extract ROI
                roi_gray = gray_frame[y:y+h, x:x+w]
                
                if roi_gray.size == 0:
                    continue
                    
                # 3. Preprocess & Predict
                roi_tensor = transform(roi_gray).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(roi_tensor)
                    # Use Softmax to get percentages instead of raw logits
                    probs = F.softmax(outputs, dim=1).squeeze().numpy()
                    
                current_probs = probs
                max_idx = np.argmax(probs)
                dominant_emotion = EMOTIONS[max_idx]
                color = EMOTION_COLORS[dominant_emotion]
                
                # 4. Add Text to Video
                cv2.putText(rgb_frame, f"{dominant_emotion} ({int(probs[max_idx]*100)}%)", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # --- Update UI Components ---
            
            # 1. Update Video Feed
            st_frame.image(rgb_frame, channels="RGB", use_column_width=True)
            
            # 2. Update Text Label
            if dominant_emotion != "Waiting...":
                current_emotion_text.markdown(
                    f"<div class='emotion-text' style='color: {EMOTION_COLORS.get(dominant_emotion, '#000')}; border: 2px solid {EMOTION_COLORS.get(dominant_emotion, '#000')}'>"
                    f"{dominant_emotion}"
                    "</div>", 
                    unsafe_allow_html=True
                )
            
            # 3. Update Bar Chart
            # We create a dataframe for the chart
            df = pd.DataFrame({
                "Emotion": EMOTIONS,
                "Probability": current_probs
            })
            # To keep colors consistent, we can't easily map colors in st.bar_chart 
            # without using Altair, but simple bar_chart is faster for loops.
            chart_placeholder.bar_chart(df.set_index("Emotion"))
            
            if stop_btn:
                break
        
        cap.release()