import streamlit as st

# MUST be first Streamlit command - before any other imports
st.set_page_config(page_title="Register Face", layout="wide")

from PIL import Image
import cv2
import numpy as np
import json
import os
from datetime import datetime
import logging
import platform

# Environment detection
IS_CLOUD = os.getenv("STREAMLIT_RUNTIME_EPHEMERAL_DISK_PATH") is not None or "streamlitcloud" in os.getcwd()

logging.basicConfig(
    filename='/tmp/face_recognition.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# Add custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #00D9FF 0%, #0099CC 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton > button:hover {
        box-shadow: 0 0 20px #00D9FF;
    }
    h1, h2 {
        background: linear-gradient(135deg, #00D9FF 0%, #00FF88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
</style>
""", unsafe_allow_html=True)

st.title("📝 Register Face")
st.write("✨ Register a new person with their face")

@st.cache_resource
def load_face_model():
    """Load InsightFace model - lazy loaded to avoid circular imports"""
    try:
        import insightface
    except ImportError:
        st.error("❌ insightface not installed. Please install: pip install insightface")
        st.stop()
    
    logger.info("Loading InsightFace model for registration...")
    analyzer = insightface.app.FaceAnalysis(ctx_id=-1, providers=['CPUExecutionProvider'])
    analyzer.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))
    return analyzer

analyzer = load_face_model()

def extract_face_embedding(image_array):
    """Extract face embedding using InsightFace"""
    faces = analyzer.get(image_array)
    
    if len(faces) == 0:
        logger.warning("No face detected in image")
        return None
    
    largest_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    logger.info("Face detected and extracted")
    return largest_face.embedding

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
                # Convert RGB to BGR for OpenCV/InsightFace
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                embedding = extract_face_embedding(image_array)
                
                if embedding is None:
                    st.error("❌ No face detected")
                else:
                    db = load_database()
                    person = {
                        "id": len(db["faces"]) + 1,
                        "name": name,
                        "relation": relation,
                        "notes": notes,
                        "embedding": embedding.tolist(),  # Store as list for JSON
                        "registered_at": datetime.now().isoformat()
                    }
                    db["faces"].append(person)
                    save_database(db)
                    logger.info(f"Registered {name}")
                    st.success(f"✅ {name} registered!")
                    st.balloons()
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Registration error: {e}")

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