# DeepFace Face Compare

This project is a simple face recognition application built using Python and the DeepFace library. It allows users to upload an image and compare it with a reference image to determine if they match.

## Project Structure

```
deepface-face-compare
├── src
│   ├── app.py                # Entry point of the application with Streamlit interface
│   └── services
│       └── face_recognition.py # Logic for face recognition using DeepFace
├── data
│   └── reference
│       └── .gitkeep          # Keeps the reference directory in version control
├── requirements.txt           # Lists the dependencies required for the project
└── README.md                  # Documentation for the project
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd deepface-face-compare
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   streamlit run src/app.py
   ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload an image you want to compare.

4. The application will process the image and display the results.

## Dependencies

- Streamlit
- DeepFace

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes.