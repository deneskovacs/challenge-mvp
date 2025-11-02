def load_reference_image(reference_path):
    from deepface import DeepFace
    import cv2

    reference_image = cv2.imread(reference_path)
    return reference_image

def compare_faces(uploaded_image, reference_image):
    from deepface import DeepFace

    result = DeepFace.verify(uploaded_image, reference_image)
    return result['verified']