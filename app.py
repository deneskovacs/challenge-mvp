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

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Password protection - only on cloud
def check_password():
    if not IS_CLOUD:
        # Skip password on local execution
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

# Add custom CSS for neon gradient
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
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00F0FF 0%, #00BBDD 100%);
        box-shadow: 0 0 20px #00D9FF;
    }
    .stSuccess {
        background: linear-gradient(135deg, #00D9FF 0%, #00FF88 100%);
        border-radius: 8px;
    }
    .stError {
        background: linear-gradient(135deg, #FF006E 0%, #FF4365 100%);
        border-radius: 8px;
    }
    .stWarning {
        background: linear-gradient(135deg, #FFB703 0%, #FB5607 100%);
        border-radius: 8px;
    }
    h1, h2, h3 {
        background: linear-gradient(135deg, #00D9FF 0%, #00FF88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
    }
    .stMetric {
        background: linear-gradient(135deg, #1a1f3a 0%, #2d2f52 100%);
        border: 2px solid #00D9FF;
        border-radius: 8px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Configure logging
logging.basicConfig(
    filename='/tmp/face_recognition.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

st.title("🔍 Face Recognition")
st.write("✨ Capture your face to see if you're recognized")

if not SKLEARN_AVAILABLE:
    st.error("❌ Required dependencies not available. Please check deployment logs.")
    st.stop()

@st.cache_resource
def load_face_model():
    """Load InsightFace model - lazy loaded to avoid circular imports"""
    try:
        import insightface
    except ImportError:
        st.error("❌ insightface not installed. Please install: pip install insightface")
        st.stop()
    
    ctx_id = -1
    logger.info(f"Loading InsightFace model (Cloud: {IS_CLOUD}, macOS: {IS_MACOS})...")
    analyzer = insightface.app.FaceAnalysis(ctx_id=ctx_id, providers=['CPUExecutionProvider'])
    analyzer.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))
    logger.info("InsightFace model prepared")
    return analyzer

analyzer = load_face_model()

def extract_face_embedding(image_array):
    """Extract face embedding using InsightFace"""
    faces = analyzer.get(image_array)
    if len(faces) == 0:
        logger.warning("No face detected in image")
        return None
    largest_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    logger.info(f"Face detected and extracted")
    return largest_face.embedding

def load_database():
    """Load face database"""
    db_path = "/tmp/face_database.json"
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            db = json.load(f)
            logger.info(f"Database loaded: {len(db['faces'])} faces")
            return db
    logger.info("Database empty, creating new")
    return {"faces": []}

def compare_embeddings(embedding1, embedding2):
    """Compare two embeddings - lazy load sklearn"""
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        st.error("❌ sklearn not installed. Please install: pip install scikit-learn")
        return 0
    
    embedding1 = np.array(embedding1).flatten()
    embedding2 = np.array(embedding2).flatten()
    
    if embedding1.shape != embedding2.shape:
        embedding1 = embedding1.reshape(1, -1)
        embedding2 = embedding2.reshape(1, -1)
    
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return max(0, (similarity + 1) / 2 * 100)

def find_match(captured_embedding, threshold=70):
    """Find matching person in database"""
    db = load_database()
    best_match = None
    best_score = 0
    
    for person in db["faces"]:
        stored_embedding = np.array(person["embedding"])
        score = compare_embeddings(captured_embedding, stored_embedding)
        
        if score > best_score:
            best_score = score
            best_match = person
    
    if best_score >= threshold:
        logger.info(f"Match found: {best_match['name']} with score {best_score:.1f}%")
        return best_match, best_score
    
    logger.info(f"No match found (best score: {best_score:.1f}%)")
    return None, best_score

def speak_text(text):
    """Text-to-speech with environment detection"""
    try:
        logger.info(f"Audio generation attempted (Cloud: {IS_CLOUD}, macOS: {IS_MACOS})")
        
        # Only attempt audio on macOS locally
        if not IS_MACOS:
            logger.warning("Audio not available (not macOS)")
            return None
        
        if IS_CLOUD:
            logger.warning("Audio not available on Streamlit Cloud")
            return None
        
        logger.info(f"Generating audio for: '{text}'")
        
        # Use macOS say command
        result = subprocess.run(
            ['say', '-o', '/tmp/speech.aiff', text],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            # Try to convert to WAV
            try:
                convert_result = subprocess.run(
                    ['ffmpeg', '-i', '/tmp/speech.aiff', '-acodec', 'pcm_s16le', '-ar', '44100', '/tmp/speech.wav', '-y'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if convert_result.returncode == 0:
                    with open('/tmp/speech.wav', 'rb') as f:
                        return f.read()
            except:
                pass
            
            # Fallback to AIFF
            with open('/tmp/speech.aiff', 'rb') as f:
                logger.info(f"Audio generated (AIFF)")
                return f.read()
        
        return None
    except Exception as e:
        logger.error(f"Audio generation error: {e}")
        return None

with st.expander("📋 How to get best results"):
    st.markdown("""
    - Clear, frontal face photo
    - Good lighting
    - No sunglasses or hats
    - Face clearly in frame
    """)

st.subheader("Capture Your Face")
captured_image = st.camera_input("Take a photo", key="camera_capture")

if captured_image:
    st.success("✓ Photo captured")
    logger.info("User captured a photo")
    
    if st.button("🔍 Recognize Face", type="primary"):
        logger.info("Recognize Face button clicked")
        with st.spinner("Analyzing..."):
            try:
                progress = st.progress(0, text="Processing image...")
                cap_array = cv2.cvtColor(np.array(Image.open(captured_image)), cv2.COLOR_RGB2BGR)
                cap_embedding = extract_face_embedding(cap_array)
                
                if cap_embedding is None:
                    st.error("❌ No face detected")
                    logger.error("No face detected in captured image")
                else:
                    progress.progress(100, text="Searching database...")
                    
                    match, score = find_match(cap_embedding, threshold=70)
                    
                    st.divider()
                    st.metric("Match Score", f"{score:.1f}%")
                    
                    if match:
                        st.success(f"✅ **Welcome {match['name']}!**")
                        st.write(f"**Relation:** {match['relation']}")
                        if match['notes']:
                            st.write(f"**Notes:** {match['notes']}")
                        logger.info(f"Recognition successful for {match['name']}")
                        
                        st.session_state.matched_person = match
                        st.session_state.show_audio = True
                    else:
                        st.warning("⚠️ No match found in database")
                        st.info("Please register first in the 'Register Face' page")
                        logger.warning("No match found in database")
                        st.session_state.show_audio = False
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Recognition error: {e}")
                st.session_state.show_audio = False

# Show audio buttons if person matched
if st.session_state.get("show_audio", False) and "matched_person" in st.session_state:
    match = st.session_state.matched_person
    
    st.divider()
    st.subheader("🔊 Audio Announcements")
    
    audio_col1, audio_col2, audio_col3 = st.columns(3)
    
    with audio_col1:
        if st.button("🔊 Say Name", use_container_width=True, key="say_name_btn"):
            logger.info(f"'Say Name' button clicked for {match['name']}")
            announcement = f"Hello {match['name']}"
            st.info(f"Speaking: \"{announcement}\"")
            audio_bytes = speak_text(announcement)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
            elif IS_CLOUD:
                st.info("ℹ️ Audio not available on cloud (works locally on macOS)")
            else:
                st.info("ℹ️ Audio not available on this system")
    
    with audio_col2:
        if st.button("👨‍👩‍👧 Say Relation", use_container_width=True, key="say_relation_btn"):
            logger.info(f"'Say Relation' button clicked for {match['name']}")
            relation_text = f"{match['name']} is my {match['relation']}"
            st.info(f"Speaking: \"{relation_text}\"")
            audio_bytes = speak_text(relation_text)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
            elif IS_CLOUD:
                st.info("ℹ️ Audio not available on cloud (works locally on macOS)")
            else:
                st.info("ℹ️ Audio not available on this system")
    
    with audio_col3:
        if st.button("👋 Say Welcome", use_container_width=True, key="say_welcome_btn"):
            logger.info(f"'Say Welcome' button clicked for {match['name']}")
            welcome_text = f"Welcome {match['name']}"
            st.info(f"Speaking: \"{welcome_text}\"")
            audio_bytes = speak_text(welcome_text)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
            elif IS_CLOUD:
                st.info("ℹ️ Audio not available on cloud (works locally on macOS)")
            else:
                st.info("ℹ️ Audio not available on this system")
else:
    if not st.session_state.get("show_audio", False):
        st.info("ℹ️ Recognize a face first to see audio options")

# Display log file link
st.divider()
if st.checkbox("📋 Show Debug Logs"):
    try:
        with open('/tmp/face_recognition.log', 'r') as f:
            logs = f.read()
            st.code(logs, language="log")
    except FileNotFoundError:
        st.info("No logs yet")
