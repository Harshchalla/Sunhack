import streamlit as st
import distinct_faces

st.title("Video Upload App")
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file is not None:
    st.video(video_file)
    # save the video file in a tmp folder
    # create a new sess id using uuid
    # call distinct_faces.main(video_filepath, sess_id)
    # get a filepath to a zip that has pngs, which represent faces. show that to user
