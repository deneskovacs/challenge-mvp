import streamlit as st
from PIL import Image
import cv2
import numpy as np
import insightface
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    filename='/tmp/face_recognition.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Register Face", layout="wide")
st.title("📝 Register Face")
st.write("Register a new person by uploading a photo and adding their information")

@st.cache_resource
def load_face_model():
    """Load InsightFace model"""
    ctx_id = -1
    logger.info("Loading InsightFace model for registration...")
    return insightface.app.FaceAnalysis(ctx_id=ctx_id, providers=['CPUExecutionProvider'])

analyzer = load_face_model()
analyzer.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))
logger.info("InsightFace model prepared for registration")

def extract_face_embedding(image_array):
    """Extract face embedding"""
    faces = analyzer.get(image_array)
    if len(faces) == 0:
        logger.warning("No face detected during registration")
        return None
    largest_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    logger.info("Face detected and embedded")
    return largest_face.embedding

def load_database():
    """Load face database"""
    db_path = "/tmp/face_database.json"
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            return json.load(f)
    return {"faces": []}

def save_database(db):
    """Save face database"""
    db_path = "/tmp/face_database.json"
    with open(db_path, 'w') as f:
        json.dump(db, f, indent=2)

# Registration form
st.subheader("Register New Person")

col1, col2 = st.columns(2)

with col1:
    st.write("**Upload Photo**")
    uploaded_file = st.file_uploader("Upload a clear photo", type=["jpg", "jpeg", "png"], key="register_upload")
    if uploaded_file:
        st.image(Image.open(uploaded_file), use_column_width=True)

with col2:
    st.write("**Person Information**")
    name = st.text_input("Full Name", placeholder="e.g., John Doe")
    relation = st.selectbox("Relation", ["Family", "Friend", "Colleague", "Other"])
    custom_relation = st.text_input("Or specify custom relation", placeholder="e.g., Daughter, Grandson")
    notes = st.text_area("Additional notes", placeholder="e.g., Birthday: Jan 15")

if st.button("✅ Register Face", type="primary"):
    if not uploaded_file:
        st.error("Please upload a photo")
    elif not name:
        st.error("Please enter a name")
    else:
        with st.spinner("Processing..."):
            try:
                logger.info(f"Registration process started for {name}")
                
                # Extract embedding
                image_array = cv2.cvtColor(np.array(Image.open(uploaded_file)), cv2.COLOR_RGB2BGR)
                embedding = extract_face_embedding(image_array)
                
                if embedding is None:
                    st.error("❌ No face detected in photo")
                    logger.error(f"Face detection failed for {name}")
                else:
                    # Save to database
                    db = load_database()
                    person = {
                        "id": len(db["faces"]) + 1,
                        "name": name,
                        "relation": custom_relation if custom_relation else relation,
                        "notes": notes,
                        "embedding": embedding.tolist(),
                        "registered_at": datetime.now().isoformat()
                    }
                    db["faces"].append(person)
                    save_database(db)
                    
                    logger.info(f"Successfully registered {name}")
                    st.success(f"✅ {name} registered successfully!")
                    st.balloons()
            except Exception as e:
                logger.error(f"Registration error: {e}")
                st.error(f"Error: {str(e)}")

st.divider()
st.subheader("Registered Persons")

db = load_database()
if db["faces"]:
    for person in db["faces"]:
        with st.expander(f"👤 {person['name']} ({person['relation']})"):
            st.write(f"**Relation:** {person['relation']}")
            st.write(f"**Notes:** {person['notes'] or 'None'}")
            st.write(f"**Registered:** {person['registered_at'][:10]}")
            
            if st.button(f"🗑️ Delete {person['name']}", key=f"delete_{person['id']}"):
                db["faces"] = [p for p in db["faces"] if p['id'] != person['id']]
                save_database(db)
                logger.info(f"Deleted {person['name']} from database")
                st.success(f"Deleted {person['name']}")
                st.rerun()
else:
    st.info("No faces registered yet")