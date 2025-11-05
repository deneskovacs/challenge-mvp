import streamlit as st

# MUST be first Streamlit command - before any other imports
st.set_page_config(page_title="Face Recognition", layout="wide")

import cv2
import numpy as np
import json
import os
import logging
from datetime import datetime
import platform
import subprocess
from PIL import Image

# Environment detection
IS_CLOUD = os.getenv("STREAMLIT_RUNTIME_EPHEMERAL_DISK_PATH") is not None or "streamlitcloud" in os.getcwd()
IS_MACOS = platform.system() == "Darwin"

# Password protection - only on cloud
def check_password():
    if not IS_CLOUD:
        return True
    
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    
    if st.session_state.password_correct:
        return True
    
    st.title("🔐 Face Recognition - Login Required")
    
    with st.form("password_form"):
        password = st.text_input("Enter password", type="password")
        if st.form_submit_button("Login"):
            if password == st.secrets.get("app_password", ""):
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("❌ Incorrect password")
    return False

if not check_password():
    st.stop()

# Configure logging
logging.basicConfig(
    filename='/tmp/face_recognition.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

st.title("🔍 Face Recognition")
st.write("✨ Simple face comparison using OpenCV")

# Use simple OpenCV face detection instead of InsightFace
@st.cache_resource
def load_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

detector = load_detector()

def extract_face_features(image_array):
    """Extract simple face features using OpenCV"""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    
    if len(faces) == 0:
        return None
    
    # Get the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y+h, x:x+w]
    
    # Resize to standard size and flatten as feature vector
    face_resized = cv2.resize(face_roi, (100, 100))
    features = face_resized.flatten()
    
    return features.tolist()

def load_database():
    """Load face database"""
    db_path = "/tmp/face_database.json"
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            return json.load(f)
    return {"faces": []}

def compare_features(features1, features2):
    """Simple feature comparison using correlation"""
    f1 = np.array(features1)
    f2 = np.array(features2)
    
    # Normalize features
    f1 = (f1 - np.mean(f1)) / (np.std(f1) + 1e-8)
    f2 = (f2 - np.mean(f2)) / (np.std(f2) + 1e-8)
    
    # Calculate correlation
    correlation = np.corrcoef(f1, f2)[0, 1]
    
    # Convert to percentage (0-100)
    return max(0, correlation * 100)

def find_match(captured_features, threshold=60):
    """Find matching person in database"""
    db = load_database()
    best_match = None
    best_score = 0
    
    for person in db["faces"]:
        stored_features = person.get("features", [])
        if not stored_features:
            continue
            
        score = compare_features(captured_features, stored_features)
        
        if score > best_score:
            best_score = score
            best_match = person
    
    if best_score >= threshold:
        return best_match, best_score
    
    return None, best_score

st.subheader("Capture Your Face")
captured_image = st.camera_input("Take a photo", key="camera_capture")

if captured_image:
    st.success("✓ Photo captured")
    
    if st.button("🔍 Recognize Face", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                # Convert image
                cap_array = cv2.cvtColor(np.array(Image.open(captured_image)), cv2.COLOR_RGB2BGR)
                cap_features = extract_face_features(cap_array)
                
                if cap_features is None:
                    st.error("❌ No face detected")
                else:
                    match, score = find_match(cap_features, threshold=60)
                    
                    st.divider()
                    st.metric("Match Score", f"{score:.1f}%")
                    
                    if match:
                        st.success(f"✅ **Welcome {match['name']}!**")
                        st.write(f"**Relation:** {match['relation']}")
                        if match.get('notes'):
                            st.write(f"**Notes:** {match['notes']}")
                    else:
                        st.warning("⚠️ No match found in database")
                        st.info("Please register first in the 'Register Face' page")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Debug logs
st.divider()
if st.checkbox("📋 Show Debug Logs"):
    try:
        with open('/tmp/face_recognition.log', 'r') as f:
            logs = f.read()
            st.code(logs, language="log")
    except FileNotFoundError:
        st.info("No logs yet")
