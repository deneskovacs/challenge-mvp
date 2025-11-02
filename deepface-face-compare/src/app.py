import streamlit as st
from services.face_recognition import compare_faces

st.title("Face Recognition App")
st.write("Upload an image to compare with the reference image.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    result = compare_faces(uploaded_file)
    if result:
        st.success("The faces match!")
    else:
        st.error("The faces do not match.")