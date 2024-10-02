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
# get 2 streamlit cols
col1, col2 = st.columns(2)
video_file = col1.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if 'emoji_path' not in st.session_state:
    st.session_state['emoji_path'] = EMOJI_PATH

emoji = col2.file_uploader('Upload a mask', type=['png', 'jpg', 'jpeg'])
# save the mask at a path
if emoji is not None:
    mask_path = 'uploaded_mask.png'
    with open(mask_path, 'wb') as f:
        f.write(emoji.getvalue())
    st.session_state['emoji_path'] = mask_path
with open(st.session_state['emoji_path'], 'rb') as f:
    # show the mask in col2
    col2.image(f.read(), use_column_width=True)

if video_file is not None:
    with col1:
        with st.spinner('Gathering faces ...'):
            faces_dir, video_hash = process_video(video_file)
    captions_list: list[str] = []
    for face_filename in os.listdir(faces_dir):
        if face_filename.endswith(".png"):
            caption = face_filename[len(video_hash)+1:]
            with open(f'{faces_dir}/{face_filename}', 'rb') as f:
                face_bytes = f.read()
            face_bytes_io = BytesIO(face_bytes)
            col1.image(face_bytes_io, caption=caption, width=200)
            captions_list.append(caption)
    selected_captions = st.multiselect("Select the distinct faces", captions_list)
    # add a submit button
    if st.button("Submit"):
        with st.spinner('Creating video ... '):
            processed_video_filepath = slap_emoji.main(st.session_state[f'video_filepath_{video_hash}'], st.session_state['emoji_path'], [f'{faces_dir}/{video_hash}_{x}' for x in selected_captions])
            st.session_state[f'processed_video_{video_hash}'] = processed_video_filepath
    if f'processed_video_{video_hash}' in st.session_state:
        output_video_path = st.session_state[f'processed_video_{video_hash}']
        ext = output_video_path.split('.')[-1]
        with open(output_video_path, 'rb') as f:
            st.download_button('Download Video', f.read(), 'processed_video.'+ext)
