import cv2
import numpy as np
import torch

from scipy.spatial.distance import cosine
from facenet_pytorch import InceptionResnetV1


resnet = InceptionResnetV1(pretrained='vggface2').eval()
print('Model loaded')


def slap_emoji_on_face(img: np.ndarray, emoji: np.ndarray, query_face: np.ndarray) -> np.ndarray:
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  face_cascade = cv2.CascadeClassifier('../pretrained_haarcascades/haarcascade_frontalface_default.xml')
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
  for face in faces:
    x, y, w, h = face
    if not is_same_face(img[y:y+h, x:x+w], query_face): continue
    resized_emoji = cv2.resize(emoji, (w, h))
    emoji_gray = cv2.cvtColor(resized_emoji, cv2.COLOR_BGR2GRAY)
    _, emoji_mask = cv2.threshold(emoji_gray, 10, 255, cv2.THRESH_BINARY)
    emoji_mask_inv = cv2.bitwise_not(emoji_mask)
    face_region = img[y:y+h, x:x+w]
    masked_face = cv2.bitwise_and(face_region, face_region, mask=emoji_mask_inv)
    masked_emoji = cv2.bitwise_and(resized_emoji, resized_emoji, mask=emoji_mask)
    combined = cv2.add(masked_face, masked_emoji)
    img[y:y+h, x:x+w] = combined
  return img

@torch.no_grad()
def extract_embeddings(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32')
    face = (face - 127.5) / 127.5
    embedding = resnet(torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float())
    return embedding

def compare_faces(embedding1, embedding2):
    return cosine(embedding1, embedding2)


THRESHOLD = 0.5
def is_same_face(curr_face: np.ndarray, anchor_face: np.ndarray) -> bool:
  curr_embeddings = extract_embeddings(curr_face)
  anchor_embeddings = extract_embeddings(anchor_face)
  return True if cosine(curr_embeddings, anchor_embeddings) <= THRESHOLD else False



def main(video_filepath: str, emoji_filepath: str, query_face_filepath: str) -> str:
  '''
  Takes in a video and query face.
  Slaps emoji on all instances of query face in the given video
  saves the modified video
  returns the modified video filepath
  '''
  cap = cv2.VideoCapture(video_filepath)
  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out_filepath = 'output_' + video_filepath
  out = cv2.VideoWriter(out_filepath, fourcc, fps, (width, height))

  emoji = cv2.imread(emoji_filepath)
  query_face = cv2.imread(query_face_filepath)
  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
      modified_frame = slap_emoji_on_face(frame, emoji, query_face)
      out.write(modified_frame)
    else:
      break

  cap.release()
  out.release()
  return out_filepath


if __name__ == '__main__':
  video_path = 'trial_video.mov'
  emoji_path = './smiling_emoji.png'
  query_face_filepath = 'query_face.jpeg'
  out_fp = main(video_path, emoji_path, query_face_filepath)
  print('Video saved at:', out_fp)
