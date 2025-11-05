import streamlit as st

st.set_page_config(page_title="Register Face", layout="wide")

from PIL import Image
import cv2
import numpy as np
import json
import os
from datetime import datetime
import logging

# Environment detection
IS_CLOUD = os.getenv("STREAMLIT_RUNTIME_EPHEMERAL_DISK_PATH") is not None or "streamlitcloud" in os.getcwd()

# Configure logging
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

st.title("📝 Register Face")

# Check if InsightFace is available
def check_insightface():
    try:
        import insightface
        return True
    except ImportError:
        return False

INSIGHTFACE_AVAILABLE = check_insightface()

if INSIGHTFACE_AVAILABLE:
    st.success("✅ InsightFace available - High accuracy mode")
else:
    st.warning("⚠️ InsightFace not installed. Using OpenCV fallback.")

def get_face_analyzer():
    """Get or create InsightFace analyzer safely"""
    if not INSIGHTFACE_AVAILABLE:
        return None
        
    if 'face_analyzer' not in st.session_state:
        try:
            import insightface
            analyzer = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
            analyzer.prepare(ctx_id=0, det_size=(640, 640))
            st.session_state.face_analyzer = analyzer
            logger.info("InsightFace analyzer loaded for registration")
            return analyzer
        except Exception as e:
            logger.error(f"Failed to load InsightFace: {e}")
            st.error(f"Failed to load InsightFace model: {e}")
            return None
    
    return st.session_state.face_analyzer

def extract_face_embedding_insightface(image_array):
    """Extract face embedding using InsightFace or OpenCV fallback"""
    if INSIGHTFACE_AVAILABLE:
        analyzer = get_face_analyzer()
        if analyzer is not None:
            try:
                faces = analyzer.get(image_array)
                if len(faces) == 0:
                    return None
                
                largest_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
                logger.info(f"Face detected with InsightFace - confidence: {largest_face.det_score:.3f}")
                return largest_face.embedding.tolist()
            except Exception as e:
                logger.error(f"InsightFace extraction failed: {e}")
    
    # Fallback to OpenCV
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        if len(faces) == 0:
            return None
        
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (100, 100))
        features = face_resized.flatten().astype(float)
        features = features / (np.linalg.norm(features) + 1e-8)
        
        logger.info("Face detected with OpenCV fallback")
        return features.tolist()
    except Exception as e:
        logger.error(f"OpenCV face detection failed: {e}")
        return None

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
        # Show image info for debugging
        img_array = np.array(Image.open(uploaded_file))
        st.caption(f"Image size: {img_array.shape}")

with col2:
    st.write("**Person Information**")
    name = st.text_input("Full Name")
    relation = st.selectbox("Relation", ["Family", "Friend", "Colleague", "Other"])
    notes = st.text_area("Notes")

if st.button("✅ Register Face"):
    if not uploaded_file or not name:
        st.error("Please fill all fields")
    else:
        method = "InsightFace" if INSIGHTFACE_AVAILABLE else "OpenCV"
        with st.spinner(f"Processing face with {method}..."):
            try:
                # Convert image with better handling
                image_pil = Image.open(uploaded_file)
                
                # Convert to RGB if needed
                if image_pil.mode != 'RGB':
                    image_pil = image_pil.convert('RGB')
                
                image_array = np.array(image_pil)
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                logger.info(f"Processing image with shape: {image_array.shape}")
                st.info(f"Processing image: {image_array.shape}")
                
                embedding = extract_face_embedding_insightface(image_array)
                
                if embedding is None:
                    st.error("❌ No face detected in uploaded image")
                    st.info("Try uploading a clearer image with a visible face")
                    
                    # Show suggestions
                    st.markdown("""
                    **InsightFace Tips:**
                    - Face should be clearly visible and front-facing
                    - Good lighting conditions
                    - No heavy shadows or glare
                    - Face should fill at least 20% of the image
                    - Avoid sunglasses, masks, or hair covering the face
                    """)
                else:
                    db = load_database()
                    person = {
                        "id": len(db["faces"]) + 1,
                        "name": name,
                        "relation": relation,
                        "notes": notes,
                        "embedding": embedding,
                        "registered_at": datetime.now().isoformat()
                    }
                    db["faces"].append(person)
                    save_database(db)
                    logger.info(f"Successfully registered: {name}")
                    st.success(f"✅ {name} registered successfully with InsightFace!")
                    st.balloons()
            except Exception as e:
                st.error(f"Registration failed: {str(e)}")
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