import hashlib
import streamlit as st

import os
import uuid
from zipfile import ZipFile
from io import BytesIO


import distinct_faces
import slap_emoji

EMOJI_PATH = './smiling_emoji.png'
def calculate_video_hash(video_file):
    """Calculate a hash of the video file"""
    hash = hashlib.sha256()
    hash.update(video_file.read())
    video_file.seek(0)  # reset the file pointer
    return hash.hexdigest()



def save_video_tmp(video_file, video_hash):
    tmp_folder = f"/tmp/{video_hash}"
    os.makedirs(tmp_folder, exist_ok=True)
    video_filepath = os.path.join(tmp_folder, video_file.name)
    with open(video_filepath, "wb") as f:
        f.write(video_file.read())
    return video_filepath


def process_video(video_file):
    """Process the video and cache the result"""
    video_hash = calculate_video_hash(video_file)
    if f"faces_dir_{video_hash}" not in st.session_state:
        video_filepath = save_video_tmp(video_file, video_hash)
        faces_dir = distinct_faces.main(video_filepath, video_hash)
        st.session_state[f"faces_dir_{video_hash}"] = faces_dir
        st.session_state[f"video_filepath_{video_hash}"] = video_filepath
    return st.session_state[f"faces_dir_{video_hash}"], video_hash

# ===
# Streamlit
# ===
st.title("Welcome to The Masquerade")
with open('play/output_trial_video.mov', 'rb') as f:
    st.video(f.read())

video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file is not None:
    faces_dir, video_hash = process_video(video_file)
    captions_list: list[str] = []
    for face_filename in os.listdir(faces_dir):
        if face_filename.endswith(".png"):
            caption = face_filename[len(video_hash)+1:]
            with open(f'{faces_dir}/{face_filename}', 'rb') as f:
                face_bytes = f.read()
            face_bytes_io = BytesIO(face_bytes)
            st.image(face_bytes_io, caption=caption, width=200)
            captions_list.append(caption)
    selected_captions = st.multiselect("Select the distinct faces", captions_list)
    # add a submit button
    if st.button("Submit"):
        processed_video_filepath = slap_emoji.main(st.session_state[f'video_filepath_{video_hash}'], EMOJI_PATH, [f'{faces_dir}/{video_hash}_{x}' for x in selected_captions])
        st.session_state[f'processed_video_{video_hash}'] = processed_video_filepath
    if f'processed_video_{video_hash}' in st.session_state:
        with open(st.session_state[f'processed_video_{video_hash}'], 'rb') as f:
            st.video(f.read())
