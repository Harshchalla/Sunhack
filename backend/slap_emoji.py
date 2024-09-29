import cv2
import numpy as np
import torch
import os

from scipy.spatial.distance import cosine
from facenet_pytorch import InceptionResnetV1


resnet = InceptionResnetV1(pretrained='vggface2').eval()
print('Model loaded')


def slap_emoji_on_face(img: np.ndarray, emoji: np.ndarray, query_face_embeddings: list[np.ndarray]) -> np.ndarray:
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  face_cascade = cv2.CascadeClassifier('pretrained_haarcascades/haarcascade_frontalface_default.xml')
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
  for query_face_embed in query_face_embeddings:
    for face in faces:
      x, y, w, h = face
      if not is_same_face(img[y:y+h, x:x+w], query_face_embed): continue
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
    embedding = resnet(torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float()).squeeze().numpy()
    return embedding

def compare_faces(embedding1, embedding2):
    return cosine(embedding1, embedding2)


THRESHOLD = 0.5
def is_same_face(curr_face: np.ndarray, anchor_embeddings: np.ndarray) -> bool:
  curr_embeddings = extract_embeddings(curr_face)
  return True if cosine(curr_embeddings, anchor_embeddings) <= THRESHOLD else False



def main(video_filepath: str, emoji_filepath: str, query_face_filepath: list[str]) -> str:
  cap = cv2.VideoCapture(video_filepath)
  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out_filepath = os.path.join(os.path.dirname(video_filepath), 'output_' + '.'.join(os.path.basename(video_filepath).split('.')[:-1]) + '.mp4')
  out = cv2.VideoWriter(out_filepath, fourcc, fps, (width, height))
  emoji = cv2.imread(emoji_filepath)
  query_faces_embeddings = [extract_embeddings(cv2.imread(x)) for x in query_face_filepath]

  buffer_size = 60
  frame_buffer = []
  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
      frame_buffer.append(frame)
      if len(frame_buffer) == buffer_size:
        modified_frames = [slap_emoji_on_face(f, emoji, query_faces_embeddings) for f in frame_buffer]
        for f in modified_frames:
          out.write(f)
        frame_buffer = []
    else:
      break



  cap.release()
  out.release()
  return out_filepath


if __name__ == '__main__':
  video_path = 'trial_video.mov'
  emoji_path = './smiling_emoji.png'
  query_face_filepath = 'query_face.jpeg'
  out_fp = main(video_path, emoji_path, [query_face_filepath])
  print('Video saved at:', out_fp)


'''
This takes way too long ... how can I make this fast?
Here are some optimizations you can make to improve the speed of your code:

1. Use a faster face detection algorithm: The Haar cascade classifier is known to be slow. You can try using a faster face detection algorithm like the Single Shot MultiBox Detector (SSD) or the Multi-Task Cascaded Convolutional Networks (MTCNN).

2. Resize the video frames: Resizing the video frames to a smaller size before processing can significantly speed up the face detection and emoji overlay process. You can resize the frames by adding the following lines of code after reading each frame:

```python
frame = cv2.resize(frame, (640, 360))
```

This will resize the frame to a width of 640 pixels and a height of 360 pixels. You can adjust these values based on your requirements.

3. Use batch processing: Instead of processing each frame individually, you can process multiple frames at once using batch processing. This can be achieved by accumulating a certain number of frames in a buffer and then processing them together. You can modify the main loop in the `main` function as follows:

```python
buffer_size = 10
frame_buffer = []

while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    frame_buffer.append(frame)
    if len(frame_buffer) == buffer_size:
      modified_frames = [slap_emoji_on_face(f, emoji, query_faces_list) for f in frame_buffer]
      for f in modified_frames:
        out.write(f)
      frame_buffer = []
  else:
    break
```

In this modified code, we accumulate frames in the `frame_buffer` until it reaches the `buffer_size`. Once the buffer is full, we process all the frames in the buffer using a list comprehension and write the modified frames to the output video. You can adjust the `buffer_size` based on your system's memory capacity and processing power.

4. Use a faster video codec: The video codec used for writing the output video can also affect the processing speed. You can try using a faster codec like 'avc1' instead of 'mp4v'. Modify the line where you create the `cv2.VideoWriter` object as follows:

```python
fourcc = cv2.VideoWriter_fourcc(*'avc1')
```

5. Parallelize the face detection and emoji overlay process: You can use parallel processing techniques to distribute the workload across multiple cores of your CPU or even use a GPU for faster processing. However, implementing parallel processing can be complex and may require significant changes to your code.

These optimizations should help improve the speed of your code. However, keep in mind that the actual speedup achieved may vary depending on various factors such as the size of the input video, the number of faces detected in each frame, and the processing power of your system.
'''
