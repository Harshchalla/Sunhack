import streamlit as st
import os

# Create an upload folder if it doesn't exist
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def main():
    st.title("Video Upload App")

    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # Save the uploaded file to the uploads folder
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display success message
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

if __name__ == "__main__":
    main()

