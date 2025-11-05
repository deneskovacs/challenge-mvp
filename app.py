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

# Check for high-accuracy face recognition libraries
def check_face_libraries():
    libraries = {"deepface": False, "face_recognition": False}
    
    try:
        import deepface
        libraries["deepface"] = True
    except ImportError:
        pass
    
    try:
        import face_recognition
        libraries["face_recognition"] = True
    except ImportError:
        pass
    
    return libraries

FACE_LIBS = check_face_libraries()

if FACE_LIBS["deepface"]:
    st.write("✨ High-accuracy face recognition using DeepFace")
    st.success("✅ DeepFace loaded successfully")
    USE_METHOD = "deepface"
elif FACE_LIBS["face_recognition"]:
    st.write("✨ High-accuracy face recognition using face_recognition")
    st.success("✅ face_recognition loaded successfully")
    USE_METHOD = "face_recognition"
else:
    st.write("✨ Face recognition using OpenCV")
    st.warning("⚠️ Install DeepFace or face_recognition for better accuracy")
    USE_METHOD = "opencv"

def get_face_analyzer():
    """Get or create InsightFace analyzer safely"""
    if not FACE_LIBS["insightface"]:
        return None
        
    if 'face_analyzer' not in st.session_state:
        try:
            import insightface
            analyzer = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
            analyzer.prepare(ctx_id=0, det_size=(640, 640))
            st.session_state.face_analyzer = analyzer
            logger.info("InsightFace analyzer loaded successfully")
            return analyzer
        except Exception as e:
            logger.error(f"Failed to load InsightFace: {e}")
            st.error(f"Failed to load InsightFace model: {e}")
            return None
    
    return st.session_state.face_analyzer

def extract_face_embedding_opencv(image_array):
    """Fallback OpenCV face detection"""
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

def extract_face_embedding_deepface(image_array):
    """Extract face embedding using DeepFace"""
    try:
        from deepface import DeepFace
        import tempfile
        
        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            temp_path = tmp_file.name
            
        # Save image
        success = cv2.imwrite(temp_path, image_array)
        if not success:
            logger.error("Failed to save temporary image")
            return None
        
        # Extract embedding using DeepFace
        embedding = DeepFace.represent(img_path=temp_path, model_name='Facenet')[0]["embedding"]
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        logger.info("Face detected and embedding extracted with DeepFace")
        return embedding
        
    except Exception as e:
        logger.error(f"DeepFace extraction failed: {e}")
        # Clean up temp file on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return None

def extract_face_embedding_face_recognition(image_array):
    """Extract face embedding using face_recognition library"""
    try:
        import face_recognition
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_image)
        if len(face_locations) == 0:
            return None
        
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        if len(face_encodings) == 0:
            return None
        
        logger.info("Face detected with face_recognition library")
        return face_encodings[0].tolist()
        
    except Exception as e:
        logger.error(f"face_recognition extraction failed: {e}")
        return None

def extract_face_embedding(image_array):
    """Extract face embedding using best available method"""
    if USE_METHOD == "deepface":
        embedding = extract_face_embedding_deepface(image_array)
        if embedding is not None:
            return embedding
    
    elif USE_METHOD == "face_recognition":
        embedding = extract_face_embedding_face_recognition(image_array)
        if embedding is not None:
            return embedding
    
    # Fallback to OpenCV
    return extract_face_embedding_opencv(image_array)

def load_database():
    """Load face database"""
    db_path = "/tmp/face_database.json"
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            return json.load(f)
    return {"faces": []}

def compare_embeddings(embedding1, embedding2):
    """Compare two face embeddings"""
    try:
        if USE_METHOD == "deepface" or FACE_LIBS["deepface"]:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                emb1 = np.array(embedding1).reshape(1, -1)
                emb2 = np.array(embedding2).reshape(1, -1)
                similarity = cosine_similarity(emb1, emb2)[0][0]
                similarity_percentage = max(0, similarity * 100)
            except ImportError:
                # Fallback if sklearn not available
                emb1 = np.array(embedding1)
                emb2 = np.array(embedding2)
                dot_product = np.dot(emb1, emb2)
                norm_a = np.linalg.norm(emb1)
                norm_b = np.linalg.norm(emb2)
                if norm_a == 0 or norm_b == 0:
                    similarity_percentage = 0
                else:
                    similarity = dot_product / (norm_a * norm_b)
                    similarity_percentage = max(0, similarity * 100)
        else:
            # OpenCV correlation fallback
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            correlation = np.corrcoef(emb1, emb2)[0, 1]
            similarity_percentage = max(0, correlation * 100)
        
        logger.info(f"Embedding comparison: {similarity_percentage:.2f}%")
        return similarity_percentage
    except Exception as e:
        logger.error(f"Embedding comparison failed: {e}")
        return 0

def find_match(captured_embedding, threshold=75):
    """Find matching person in database"""
    db = load_database()
    best_match = None
    best_score = 0
    
    for person in db["faces"]:
        stored_embedding = person.get("embedding", [])
        if not stored_embedding:
            continue
            
        score = compare_embeddings(captured_embedding, stored_embedding)
        
        if score > best_score:
            best_score = score
            best_match = person
    
    if best_score >= threshold:
        logger.info(f"Match found: {best_match['name']} with score {best_score:.1f}%")
        return best_match, best_score
    
    logger.info(f"No match found (best score: {best_score:.1f}%)")
    return None, best_score

st.subheader("Capture Your Face")
captured_image = st.camera_input("Take a photo", key="camera_capture")

if captured_image:
    st.success("✓ Photo captured")
    
    # Show which method will be used
    if USE_METHOD == "deepface":
        st.info("🎯 Using DeepFace for high-accuracy recognition")
    elif USE_METHOD == "face_recognition":
        st.info("🎯 Using face_recognition for high-accuracy recognition")
    else:
        st.info("📷 Using OpenCV for basic recognition")
    
    if st.button("🔍 Recognize Face", type="primary"):
        with st.spinner(f"Analyzing face with {USE_METHOD}..."):
            try:
                # Convert image
                cap_array = cv2.cvtColor(np.array(Image.open(captured_image)), cv2.COLOR_RGB2BGR)
                cap_embedding = extract_face_embedding(cap_array)
                
                if cap_embedding is None:
                    st.error("❌ No face detected in image")
                    st.info("Make sure your face is clearly visible and well-lit")
                else:
                    match, score = find_match(cap_embedding, threshold=60 if USE_METHOD == "opencv" else 75)
                    
                    st.divider()
                    st.metric("Match Score", f"{score:.1f}%")
                    
                    if match:
                        st.success(f"✅ **Welcome {match['name']}!**")
                        st.write(f"**Relation:** {match['relation']}")
                        if match.get('notes'):
                            st.write(f"**Notes:** {match['notes']}")
                        logger.info(f"Recognition successful: {match['name']} ({score:.1f}%)")
                    else:
                        st.warning("⚠️ No match found in database")
                        st.info("Please register first in the 'Register Face' page")
                        logger.info(f"No match found. Best score: {score:.1f}%")
                        
            except Exception as e:
                st.error(f"Recognition failed: {str(e)}")
                logger.error(f"Recognition error: {e}")

# Debug logs
st.divider()
if st.checkbox("📋 Show Debug Logs"):
    try:
        with open('/tmp/face_recognition.log', 'r') as f:
            logs = f.read()
            st.code(logs, language="log")
    except FileNotFoundError:
        st.info("No logs yet")
