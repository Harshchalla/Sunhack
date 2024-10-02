import asyncio
import cv2
import numpy as np
import threading
from sklearn.cluster import DBSCAN
from queue import Queue

import torch
from facenet_pytorch import InceptionResnetV1
resnet = InceptionResnetV1(pretrained='vggface2').eval()
print('Model loaded')


def resize_frame(frame: np.ndarray, short_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    aspect_ratio = w/h
    if aspect_ratio > 1:
        new_w = int(aspect_ratio * short_side)
        new_h = short_side
    else:
        new_w = short_side
        new_h = int(short_side / aspect_ratio)
    return cv2.resize(frame, (new_w, new_h))


@torch.no_grad()
def extract_embeddings(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32')
    face = (face - 127.5) / 127.5
    embedding = resnet(torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float()).squeeze().numpy()
    return embedding

def main(video_path: str, sess_id: str) -> str:
  faces_list, frame_hw = get_all_faces(video_path)
  pruned_faces_list: list[np.ndarray] = prune_face_list(faces_list, frame_hw)
  face_embeddings: np.ndarray = get_face_embeddings(pruned_faces_list)
  clusters: list[list[int]] = cluster_faces(face_embeddings)  # does clustering using DBSCAN and returns indices of faces within cluster
  distinct_faces: list[np.ndarray] = get_cluster_centroids(clusters, face_embeddings, pruned_faces_list)
  distinct_faces_dir: str = save_faces(distinct_faces, sess_id)
  all_faces_dir: str = save_faces_zip(faces_list, sess_id)
  return distinct_faces_dir

def get_all_faces(video_path: str) -> tuple[list[np.ndarray], tuple[int, int]]:
    face_cascade = cv2.CascadeClassifier('pretrained_haarcascades/haarcascade_frontalface_default.xml')
    video = cv2.VideoCapture(video_path)
    faces_list = []
    frame_hw = None
    every_nth_frame = 1
    cntr = -1

    BUFFER_SIZE = 1200
    def process_frame(frame: np.ndarray) -> list[np.ndarray]:
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
      ret = [frame[y:y+h, x:x+w] for x, y, w, h in faces]
      return ret


    frame_buffer = []
    while True:
        ret, frame = video.read()
        cntr += 1
        if not ret: break
        if (cntr % every_nth_frame) == 0:  # only select every nth frame
          if frame_hw is None:
            frame_hw = tuple(frame.shape[:2])

          if len(frame_buffer) < BUFFER_SIZE:
            frame_buffer.append(resize_frame(frame, 256))
          else:
            for frame in frame_buffer:
              ret = process_frame(frame)
              if len(ret)> 0:
                faces_list.extend(ret)
            frame_buffer = []

    if len(frame_buffer):
      for frame in frame_buffer:
        ret = process_frame(frame)
        if len(ret)> 0:
          faces_list.extend(ret)
      frame_buffer = []
    video.release()
    return faces_list, frame_hw



def prune_face_list(faces_list: list[np.ndarray], frame_hw: tuple[int, int]) -> list[np.ndarray]:
  min_area = (frame_hw[0] * frame_hw[1]) / 256
  return [face for face in faces_list if (len(face) * len(face[0])) >= min_area]

def get_face_embeddings(faces_list: list[np.ndarray]) -> np.ndarray:
    return np.array([extract_embeddings(face) for face in faces_list])


def cluster_faces(face_embeddings: np.ndarray) -> list[list[int]]:
    dbscan = DBSCAN(eps=0.3, min_samples=3, metric='cosine')
    labels = dbscan.fit_predict(face_embeddings)
    clusters = [[] for _ in range(max(labels) + 1)]
    for i, label in enumerate(labels):
        if label != -1:
            clusters[label].append(i)
    return clusters

def get_cluster_centroids(clusters: list[list[int]], face_embeddings: np.ndarray, pruned_faces_list: list[np.ndarray]) -> list[np.ndarray]:
    distinct_faces = []
    for cluster in clusters:
        centroid_embedding = np.mean(face_embeddings[cluster], axis=0)
        min_distance = float('inf')
        centroid_face = None
        for face_idx in cluster:
            distance = np.linalg.norm(face_embeddings[face_idx] - centroid_embedding)
            if distance < min_distance:
                min_distance = distance
                centroid_face = pruned_faces_list[face_idx]
        distinct_faces.append(centroid_face)
    return distinct_faces



def save_faces(distinct_faces: list[np.ndarray], sess_id: str) -> str:
    import os
    from pathlib import Path

    folder_path = f'/tmp/{sess_id}'
    os.makedirs(folder_path, exist_ok=True)
    for i, face in enumerate(distinct_faces):
        face_path = f'{folder_path}/{sess_id}_{i}.png'
        cv2.imwrite(face_path, face)
    
    return folder_path

def save_faces_zip(faces_list: list[np.ndarray], sess_id: str) -> str:
    import os
    import zipfile
    from pathlib import Path

    folder_path = f'/tmp/{sess_id}_all'
    os.makedirs(folder_path, exist_ok=True)
    for i, face in enumerate(faces_list):
        face_path = f'{folder_path}/{sess_id}_{i}.png'
        cv2.imwrite(face_path, face)

    zip_file_path = f'/tmp/all_faces.zip'
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for face_path in Path(folder_path).glob('*.png'):
            zipf.write(face_path, face_path.name)

    print('Saved all extracted faces at:', zip_file_path)
    return zip_file_path

    return zip_file_path
if __name__ == '__main__':
  video_path = './trial_video.mov'
  sess_id = 'test'
  distinct_imgs_zip = main(video_path, sess_id)
