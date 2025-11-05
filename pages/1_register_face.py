import streamlit as st

st.set_page_config(page_title="Register Face", layout="wide")

from PIL import Image
import cv2
import numpy as np
import json
import os
from datetime import datetime

# Environment detection
IS_CLOUD = os.getenv("STREAMLIT_RUNTIME_EPHEMERAL_DISK_PATH") is not None or "streamlitcloud" in os.getcwd()

# Password protection - only on cloud
if IS_CLOUD:
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    
    if not st.session_state.password_correct:
        st.title("🔐 Login Required")
        with st.form("password_form"):
            password = st.text_input("Enter password", type="password")
            if st.form_submit_button("Login"):
                if password == st.secrets.get("app_password", ""):
                    st.session_state.password_correct = True
                    st.rerun()
                else:
                    st.error("❌ Incorrect password")
        st.stop()

st.title("📝 Register Face")

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
    
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (100, 100))
    features = face_resized.flatten()
    
    return features.tolist()

def load_database():
    db_path = "/tmp/face_database.json"
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            return json.load(f)
    return {"faces": []}

def save_database(db):
    with open("/tmp/face_database.json", 'w') as f:
        json.dump(db, f, indent=2)

col1, col2 = st.columns(2)

with col1:
    st.write("**Upload Photo**")
    uploaded_file = st.file_uploader("Upload a clear photo", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(Image.open(uploaded_file))

with col2:
    st.write("**Person Information**")
    name = st.text_input("Full Name")
    relation = st.selectbox("Relation", ["Family", "Friend", "Colleague", "Other"])
    notes = st.text_area("Notes")

if st.button("✅ Register Face"):
    if not uploaded_file or not name:
        st.error("Please fill all fields")
    else:
        with st.spinner("Processing..."):
            try:
                image_array = np.array(Image.open(uploaded_file))
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                features = extract_face_features(image_array)
                
                if features is None:
                    st.error("❌ No face detected")
                else:
                    db = load_database()
                    person = {
                        "id": len(db["faces"]) + 1,
                        "name": name,
                        "relation": relation,
                        "notes": notes,
                        "features": features,
                        "registered_at": datetime.now().isoformat()
                    }
                    db["faces"].append(person)
                    save_database(db)
                    st.success(f"✅ {name} registered!")
                    st.balloons()
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.divider()
st.subheader("Registered Persons")
db = load_database()
if db["faces"]:
    for person in db["faces"]:
        with st.expander(f"👤 {person['name']} ({person['relation']})"):
            st.write(f"**Relation:** {person['relation']}")
            st.write(f"**Notes:** {person['notes'] or 'None'}")
            if st.button(f"🗑️ Delete", key=f"delete_{person['id']}"):
                db["faces"] = [p for p in db["faces"] if p['id'] != person['id']]
                save_database(db)
                st.success(f"Deleted {person['name']}")
                st.rerun()
else:
    st.info("No faces registered")