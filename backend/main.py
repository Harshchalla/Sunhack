import distinct_faces
import hashlib
import streamlit as st

import os
import uuid
from zipfile import ZipFile
from io import BytesIO

def calculate_video_hash(video_file):
    """Calculate a hash of the video file"""
    hash = hashlib.sha256()
    hash.update(video_file.read())
    video_file.seek(0)  # reset the file pointer
    return hash.hexdigest()



def save_video_tmp(video_file):
    sess_id = str(uuid.uuid4())
    tmp_folder = f"/tmp/{sess_id}"
    os.makedirs(tmp_folder, exist_ok=True)
    video_filepath = os.path.join(tmp_folder, video_file.name)
    with open(video_filepath, "wb") as f:
        f.write(video_file.read())
    return video_filepath, sess_id


def process_video(video_file):
    """Process the video and cache the result"""
    video_hash = calculate_video_hash(video_file)
    if f"faces_zipfilepath_{video_hash}" not in st.session_state:
        video_filepath, sess_id = save_video_tmp(video_file)
        faces_zipfilepath = distinct_faces.main(video_filepath, sess_id)
        st.session_state[f"faces_zipfilepath_{video_hash}"] = faces_zipfilepath
    return st.session_state[f"faces_zipfilepath_{video_hash}"]

# ===
# Streamlit
# ===
st.title("Video Upload App")
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file is not None:
    faces_zipfilepath = process_video(video_file)
    captions_list: list[str] = []
    with ZipFile(faces_zipfilepath, 'r') as zip_ref:
        for face_filename in zip_ref.namelist():
            if face_filename.endswith(".png"):
                caption = face_filename[len(sess_id)+1:]
                face_bytes = zip_ref.read(face_filename)
                face_bytes_io = BytesIO(face_bytes)
                st.image(face_bytes_io, caption=caption, width=200)
                captions_list.append(caption)
    selected_captions = st.multiselect("Select the distinct faces", captions_list)

