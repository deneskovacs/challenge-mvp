# Face Recognition App

A Streamlit-based face recognition system with registration and identification capabilities.

## Features

- 📷 **Face Registration** - Upload photos to register new people
- 🔍 **Face Recognition** - Real-time camera recognition
- 👥 **Multi-person Database** - Store multiple faces with metadata
- 🔐 **Password Protection** - Secure cloud deployment
- 🎨 **Modern UI** - Neon gradient theme
- 📱 **Cross-platform** - Works on desktop and mobile

## Technology Stack

- **OpenCV** - Face detection and feature extraction
- **Streamlit** - Web interface
- **NumPy** - Numerical computations
- **scikit-learn** - Similarity calculations

## Quick Start

```bash
# Clone and setup
git clone <your-repo-url>
cd streamlit_fix

# Create virtual environment (Python 3.12 recommended)
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Usage

1. **Register faces** - Go to "Register Face" page and upload clear photos
2. **Test recognition** - Use camera on main page to test recognition
3. **View logs** - Enable debug logs to see performance metrics

## Future Improvements

- **InsightFace Integration** - For 98%+ accuracy (requires Python 3.12 and compatible ONNX Runtime)
- **Audio Announcements** - Text-to-speech on recognition
- **Advanced Analytics** - Recognition statistics and performance metrics

## Deployment

Ready for Streamlit Cloud deployment with automatic fallbacks and error handling.
