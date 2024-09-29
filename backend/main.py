import streamlit as st
import distinct_faces

import os
import uuid
from zipfile import ZipFile
from io import BytesIO

def save_video_tmp(video_file):
    sess_id = str(uuid.uuid4())
    tmp_folder = f"/tmp/{sess_id}"
    os.makedirs(tmp_folder, exist_ok=True)
    video_filepath = os.path.join(tmp_folder, video_file.name)
    with open(video_filepath, "wb") as f:
        f.write(video_file.read())
    return video_filepath, sess_id


# ===
# Streamlit
# ===
st.title("Video Upload App")
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])


if video_file is not None:
    # save the video file in a tmp folder
    # create a new sess id using uuid
    # call distinct_faces.main(video_filepath, sess_id)
    video_filepath, sess_id = save_video_tmp(video_file)
    faces_zipfilepath = distinct_faces.main(video_filepath, sess_id)
    # read and show all the faces png that are present in the zip filepath
    captions_list: list[str] = []
    with ZipFile(faces_zipfilepath, 'r') as zip_ref:
        for face_filename in zip_ref.namelist():
            if face_filename.endswith(".png"):
                # Create a BytesIO object from the face image data
                # face_filename is sess_id _ <face_num>.png
                caption = face_filename[len(sess_id)+1:]
                face_bytes = zip_ref.read(face_filename)
                face_bytes_io = BytesIO(face_bytes)
                st.image(face_bytes_io, caption=caption, width=200)
                captions_list.append(caption)

    # ask user for multiple selection. The options are captions
    selected_captions = st.multiselect("Select the distinct faces", captions_list)

    # clean up the tmp folder after the user has downloaded the selected faces or moved on to the next page
    if st.button("Clean Up"):
        if os.path.exists(tmp_folder):
            os.remove(faces_zipfilepath)
            os.rmdir(tmp_folder)
            st.success("Temporary files cleaned up.")
