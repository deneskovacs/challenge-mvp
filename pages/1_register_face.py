import streamlit as st
from PIL import Image
import cv2
import numpy as np
import json
import os
from datetime import datetime
import logging

logging.basicConfig(
    filename='/tmp/face_recognition.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Register Face", layout="wide")
st.title("📝 Register Face")

@st.cache_resource
def load_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

detector = load_detector()

def extract_face_histogram(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    
    if len(faces) == 0:
        return None
    
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y+h, x:x+w]
    hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
    return cv2.normalize(hist, hist).flatten().tolist()

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
                hist = extract_face_histogram(image_array)
                
                if hist is None:
                    st.error("❌ No face detected")
                else:
                    db = load_database()
                    person = {
                        "id": len(db["faces"]) + 1,
                        "name": name,
                        "relation": relation,
                        "notes": notes,
                        "histogram": hist,
                        "registered_at": datetime.now().isoformat()
                    }
                    db["faces"].append(person)
                    save_database(db)
                    logger.info(f"Registered {name}")
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