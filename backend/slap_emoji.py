import cv2
import numpy as np
import torch
import os

from facenet_pytorch import InceptionResnetV1
from moviepy.editor import VideoFileClip
from scipy.spatial.distance import cosine


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


face_cascade = cv2.CascadeClassifier('pretrained_haarcascades/haarcascade_frontalface_default.xml')
def slap_emoji_on_face(img: np.ndarray, emoji: np.ndarray, query_face_embeddings: list[np.ndarray]) -> np.ndarray:
    img_copy = img.copy()  # Create a writable copy of the frame
    resized_frame = resize_frame(img_copy, 160)
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for query_face_embed in query_face_embeddings:
        for face in faces:
            x, y, w, h = face
            if not is_same_face(resized_frame[y:y+h, x:x+w], query_face_embed):
                continue

            full_y, full_x = img_copy.shape[:2]
            reduced_y, reduced_x = resized_frame.shape[:2]

            new_x = int(x * full_x / reduced_x)
            new_y = int(y * full_y / reduced_y)
            new_w = int(w * full_x / reduced_x)
            new_h = int(h * full_x / reduced_x)

            resized_emoji = cv2.resize(emoji, (new_w, new_h))
            emoji_gray = cv2.cvtColor(resized_emoji, cv2.COLOR_BGR2GRAY)
            _, emoji_mask = cv2.threshold(emoji_gray, 10, 255, cv2.THRESH_BINARY)
            emoji_mask_inv = cv2.bitwise_not(emoji_mask)
            face_region = img_copy[new_y:new_y+new_h, new_x:new_x+new_w]
            masked_face = cv2.bitwise_and(face_region, face_region, mask=emoji_mask_inv)
            masked_emoji = cv2.bitwise_and(resized_emoji, resized_emoji, mask=emoji_mask)
            combined = cv2.add(masked_face, masked_emoji)
            img_copy[new_y:new_y+new_h, new_x:new_x+new_w] = combined

    return img_copy

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
    # Load the emoji and query face embeddings
    emoji = cv2.imread(emoji_filepath)
    query_faces_embeddings = [extract_embeddings(cv2.imread(x)) for x in query_face_filepath]

    # Load the original video
    video = VideoFileClip(video_filepath)

    # Process the video frames
    processed_video = video.fl_image(lambda frame: slap_emoji_on_face(frame, emoji, query_faces_embeddings))
    
    # Determine output filepath
    out_filepath = os.path.join(
        os.path.dirname(video_filepath),
        'output_' + os.path.basename(video_filepath)
    )

    # Extract the codec from the original file extension
    file_extension = os.path.splitext(video_filepath)[1].lower()
    codec = 'libx264'  # Default codec for most cases

    # Set codec based on the file extension
    if file_extension == '.mov':
        codec = 'prores'
    elif file_extension == '.avi':
        codec = 'rawvideo'  # or another suitable codec

    # Save processed video with original audio
    processed_video.set_audio(video.audio).write_videofile(out_filepath, codec=codec)

    return out_filepath


if __name__ == '__main__':
  video_path = 'trial_video.mov'
  emoji_path = './smiling_emoji.png'
  query_face_filepath = 'query_face.jpeg'
  out_fp = main(video_path, emoji_path, [query_face_filepath])
  print('Video saved at:', out_fp)



'''
cv2 Video writer is buggy. If the videwo is long, it doesn't show that start frames. It randomly snaps the frame to some timestamp. wait till tahat ts appears, and then plays.
  Also the original audio is not set. the processed video is mute.
The issue you're encountering is likely related to buffering and codec support. Here are some suggestions to resolve it:

1. **Buffer Management**: Ensure that frames are processed and written in a timely manner. If buffering is too aggressive, it might cause issues. Try reducing the buffer size to see if it helps.

2. **Audio Handling**: To include audio in the processed video, you'll need to separately handle audio tracks since OpenCV doesn't process audio. Consider using `ffmpeg` to extract and reattach audio:

   - Extract audio from the original video:
     ```
     ffmpeg -i trial_video.mov -q:a 0 -map a audio.mp3
     ```

   - Combine audio with the processed video:
     ```
     ffmpeg -i output_trial_video.mp4 -i audio.mp3 -c copy -map 0:v:0 -map 1:a:0 final_output.mp4
     ```

3. **Codec Compatibility**: Ensure that the codec used by OpenCV's `VideoWriter` is compatible with the video player. Sometimes, `mp4v` might not be ideal. Use `XVID` or another codec if needed.

4. **Alternative Libraries**: Consider using libraries like `moviepy` or `ffmpeg-python` for more robust video processing, especially for tasks that involve audio or complex video manipulations.

Here is a quick adjustment to your code to reduce buffer size:

```python
buffer_size = 300  # Reduced buffer size
```

And remember to re-encode audio separately using `ffmpeg` as suggested above.

show me example with moviepy for writing the new video frames

Certainly! `MoviePy` can be used to handle video and audio more robustly than OpenCV. Below is an example of how you can use `MoviePy` to process and save your video frames, including handling the audio track:

```python
from moviepy.editor import VideoFileClip, AudioFileClip
import numpy as np

def process_frame(frame: np.ndarray, emoji: np.ndarray, query_face_embeddings: list[np.ndarray]) -> np.ndarray:
    # Your slap_emoji_on_face function can be adapted here
    return slap_emoji_on_face(frame, emoji, query_face_embeddings)

def main_with_moviepy(video_filepath: str, emoji_filepath: str, query_face_filepath: list[str]) -> str:
    # Load the emoji and query face embeddings
    emoji = cv2.imread(emoji_filepath)
    query_faces_embeddings = [extract_embeddings(cv2.imread(x)) for x in query_face_filepath]

    # Load the original video
    video = VideoFileClip(video_filepath)
    
    # Process the video frames
    processed_video = video.fl_image(lambda frame: process_frame(frame, emoji, query_faces_embeddings))
    
    # Extract audio and save processed video
    audio = video.audio
    out_filepath = os.path.join(os.path.dirname(video_filepath), 'output_' + '.'.join(os.path.basename(video_filepath).split('.')[:-1]) + '.mp4')
    processed_video.set_audio(audio).write_videofile(out_filepath, codec='libx264')

    return out_filepath

if __name__ == '__main__':
    video_path = 'trial_video.mov'
    emoji_path = './smiling_emoji.png'
    query_face_filepath = 'query_face.jpeg'
    out_fp = main_with_moviepy(video_path, emoji_path, [query_face_filepath])
    print('Video saved at:', out_fp)
```

### Key Points:
- **`fl_image` Function**: This is a convenient way to apply a function to each frame of the video.
- **Audio Handling**: By setting the audio track from the original video onto the processed video, you maintain the original audio.
- **Codec**: `MoviePy` uses `libx264` by default for `.mp4` files, which is usually compatible with most players.

Ensure you have `MoviePy` installed in your environment:

```bash
pip install moviepy
```

This script will process each frame, apply your frame manipulation logic, and then save the video with its audio track intact.


I want to save the video in the same format i received it


To save the processed video in the same format as the original using `MoviePy`, you can utilize `ffmpeg` to determine the original video's codec and then instruct `MoviePy` to use that same codec when writing the new video. Here's how you might modify the script to achieve this:

```python
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
import os

def process_frame(frame: np.ndarray, emoji: np.ndarray, query_face_embeddings: list[np.ndarray]) -> np.ndarray:
    return slap_emoji_on_face(frame, emoji, query_face_embeddings)

def main_with_moviepy(video_filepath: str, emoji_filepath: str, query_face_filepath: list[str]) -> str:
    # Load the emoji and query face embeddings
    emoji = cv2.imread(emoji_filepath)
    query_faces_embeddings = [extract_embeddings(cv2.imread(x)) for x in query_face_filepath]

    # Load the original video
    video = VideoFileClip(video_filepath)

    # Process the video frames
    processed_video = video.fl_image(lambda frame: process_frame(frame, emoji, query_faces_embeddings))
    
    # Determine output filepath
    out_filepath = os.path.join(
        os.path.dirname(video_filepath),
        'output_' + os.path.basename(video_filepath)
    )

    # Extract the codec from the original file extension
    file_extension = os.path.splitext(video_filepath)[1].lower()
    codec = 'libx264'  # Default codec for most cases

    # Set codec based on the file extension
    if file_extension == '.mov':
        codec = 'prores'
    elif file_extension == '.avi':
        codec = 'rawvideo'  # or another suitable codec

    # Save processed video with original audio
    processed_video.set_audio(video.audio).write_videofile(out_filepath, codec=codec)

    return out_filepath

if __name__ == '__main__':
    video_path = 'trial_video.mov'
    emoji_path = './smiling_emoji.png'
    query_face_filepath = 'query_face.jpeg'
    out_fp = main_with_moviepy(video_path, emoji_path, [query_face_filepath])
    print('Video saved at:', out_fp)
```

### Key Adjustments:
- **File Extension Check**: Determine the codec based on the video file extension. You may need to adjust the codec based on your specific requirements and ensure that `ffmpeg` supports the codec for writing.
- **Preserve Format**: Using the same extension and an appropriate codec helps maintain the original format.
- **`MoviePy` Limitations**: Be aware that `MoviePy` has its own set of supported codecs which are reliant on `ffmpeg`. Ensure that the codecs you choose are available in your `ffmpeg` installation.

This approach gives you control over the codec based on the file type and can help maintain compatibility with the original format if `ffmpeg` and `MoviePy` support it.



    processed_video.set_audio(video.audio).write_videofile(out_filepath, codec=codec)

File "/home/ubuntu/Sunhack/backend/slap_emoji.py", line 84, in <lambda>
    processed_video = video.fl_image(lambda frame: slap_emoji_on_face(frame, emoji, query_faces_embeddings))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/ubuntu/Sunhack/backend/slap_emoji.py", line 53, in slap_emoji_on_face
    img[new_y:new_y+new_h, new_x:new_x+new_w] = combined

ValueError: assignment destination is read-only



The error you're encountering is due to the fact that the image frames extracted using `MoviePy` are read-only. You need to make a copy of the frame before modifying it. Here's how you can fix this:

```python
def slap_emoji_on_face(img: np.ndarray, emoji: np.ndarray, query_face_embeddings: list[np.ndarray]) -> np.ndarray:
    img_copy = img.copy()  # Create a writable copy of the frame
    resized_frame = resize_frame(img_copy, 160)
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for query_face_embed in query_face_embeddings:
        for face in faces:
            x, y, w, h = face
            if not is_same_face(resized_frame[y:y+h, x:x+w], query_face_embed):
                continue

            full_y, full_x = img_copy.shape[:2]
            reduced_y, reduced_x = resized_frame.shape[:2]

            new_x = int(x * full_x / reduced_x)
            new_y = int(y * full_y / reduced_y)
            new_w = int(w * full_x / reduced_x)
            new_h = int(h * full_x / reduced_x)

            resized_emoji = cv2.resize(emoji, (new_w, new_h))
            emoji_gray = cv2.cvtColor(resized_emoji, cv2.COLOR_BGR2GRAY)
            _, emoji_mask = cv2.threshold(emoji_gray, 10, 255, cv2.THRESH_BINARY)
            emoji_mask_inv = cv2.bitwise_not(emoji_mask)
            face_region = img_copy[new_y:new_y+new_h, new_x:new_x+new_w]
            masked_face = cv2.bitwise_and(face_region, face_region, mask=emoji_mask_inv)
            masked_emoji = cv2.bitwise_and(resized_emoji, resized_emoji, mask=emoji_mask)
            combined = cv2.add(masked_face, masked_emoji)
            img_copy[new_y:new_y+new_h, new_x:new_x+new_w] = combined

    return img_copy
```

### Key Changes:
- **Copy the Frame**: Before making any changes, create a copy of the frame using `img.copy()`. This allows you to modify the copy while keeping the original frame intact and writable.
- **Use the Copy**: Replace operations on `img` with `img_copy`.

This should resolve the `ValueError` by ensuring that you are working on a writable frame.

There is no audio
To ensure the audio is retained in the processed video, you should verify that the original video contains an audio track and that MoviePy is set up correctly to copy that audio track to the new video. Here are a few steps to troubleshoot and ensure audio is included:

1. **Verify Original Audio**: Check that the original video file (`trial_video.mov`) has an audio track. You can do this by playing the video in a media player.

2. **Ensure Correct Audio Handling**: Make sure you're setting the audio in MoviePy correctly:

   ```python
   processed_video = video.fl_image(lambda frame: process_frame(frame, emoji, query_faces_embeddings))
   processed_video = processed_video.set_audio(video.audio)
   ```

3. **Check Codec Support**: Ensure that the codec used supports audio tracks. `libx264` generally supports audio, but check if the container format (e.g., `.mov`) requires a specific codec for audio.

4. **Install FFmpeg**: Make sure FFmpeg is installed and properly configured since MoviePy relies on it for video and audio processing. You can install it via your package manager or download it from the FFmpeg website.

5. **Debug with MoviePy**: Enable verbose logging in MoviePy to see if there are any warnings or errors related to audio:

   ```python
   processed_video.set_audio(video.audio).write_videofile(out_filepath, codec=codec, verbose=True, logger='bar')
   ```

If all of the above are set correctly and you still don't hear audio, try converting the video to a more common format like `.mp4` with audio to see if the issue persists. This can help identify if the problem is with the final video format or the processing steps.


This is too slow, can't we do it in some sort of frame buffer?


To improve performance, you can indeed use a buffer to process frames in batches rather than individually. However, `MoviePy` processes video one frame at a time by default. To implement a frame buffer, you'd need to manage frame batches manually, which can be complex. Here’s a simpler optimization using a frame buffer approach:

1. **Buffer Processing**: Group frames into batches and process them together. However, ensure that video processing (especially if it's dependent on sequential frames) aligns with this approach.

2. **Parallel Processing**: Use multi-threading or multi-processing to handle frame processing in parallel.

Let's focus on a parallel processing approach:

```python
from moviepy.editor import VideoFileClip
from multiprocessing import Pool
import cv2
import numpy as np
import os

def slap_emoji_on_face_buffer(frames, emoji, query_face_embeddings):
    processed_frames = []
    for frame in frames:
        processed_frame = slap_emoji_on_face(frame, emoji, query_face_embeddings)
        processed_frames.append(processed_frame)
    return processed_frames

def process_frame(frame: np.ndarray, emoji: np.ndarray, query_face_embeddings: list[np.ndarray]) -> np.ndarray:
    return slap_emoji_on_face(frame, emoji, query_face_embeddings)

def main_with_moviepy(video_filepath: str, emoji_filepath: str, query_face_filepath: list[str]) -> str:
    emoji = cv2.imread(emoji_filepath)
    query_faces_embeddings = [extract_embeddings(cv2.imread(x)) for x in query_face_filepath]
    
    video = VideoFileClip(video_filepath)
    out_filepath = os.path.join(
        os.path.dirname(video_filepath),
        'output_' + os.path.basename(video_filepath)
    )
    codec = 'libx264'  # Default, adjust as needed

    # Convert video frames into numpy arrays for processing
    frames = list(video.iter_frames(fps=video.fps, dtype="uint8"))

    # Use multiprocessing to speed up frame processing
    num_processors = os.cpu_count()
    chunk_size = len(frames) // num_processors

    with Pool(processes=num_processors) as pool:
        processed_frames = pool.starmap(
            slap_emoji_on_face_buffer, 
            [(frames[i:i + chunk_size], emoji, query_faces_embeddings) for i in range(0, len(frames), chunk_size)]
        )

    # Flatten the list of lists
    processed_frames = [frame for sublist in processed_frames for frame in sublist]

    # Convert processed frames back to a video
    processed_video = video.set_frames(processed_frames)

    # Write the video with the original audio
    processed_video.set_audio(video.audio).write_videofile(out_filepath, codec=codec)

    return out_filepath

if __name__ == '__main__':
    video_path = 'trial_video.mov'
    emoji_path = './smiling_emoji.png'
    query_face_filepath = 'query_face.jpeg'
    out_fp = main_with_moviepy(video_path, emoji_path, [query_face_filepath])
    print('Video saved at:', out_fp)
```

### Key Points:
- **Multiprocessing**: Utilize `multiprocessing.Pool` to process frames in parallel, which can significantly speed up processing.
- **Frame Buffering**: Convert video frames into a list for buffered processing. This approach can be adapted to batch frames.
- **Chunking**: Divide frames into chunks based on CPU count for parallel processing.

Keep in mind that video processing is inherently intensive and improvements may vary based on system capabilities and video complexity. This approach should enhance performance by leveraging parallel processing.



Give me buffer processing approach

To implement a buffer processing approach, we'll process frames in chunks (buffers) instead of individually. This reduces overhead and can improve performance for batch operations. Below is a simple example using a frame buffer:

```python
from moviepy.editor import VideoFileClip, concatenate_videoclips
import cv2
import numpy as np
import os

def slap_emoji_on_face_buffer(frames, emoji, query_face_embeddings):
    processed_frames = []
    for frame in frames:
        processed_frame = slap_emoji_on_face(frame, emoji, query_face_embeddings)
        processed_frames.append(processed_frame)
    return processed_frames

def process_buffered_frames(video_filepath, emoji, query_face_embeddings, buffer_size):
    video = VideoFileClip(video_filepath)
    frames = list(video.iter_frames(fps=video.fps, dtype='uint8'))
    clips = []

    for i in range(0, len(frames), buffer_size):
        buffer = frames[i:i + buffer_size]
        processed_buffer = slap_emoji_on_face_buffer(buffer, emoji, query_face_embeddings)
        clip = VideoFileClip(np.array(processed_buffer), fps=video.fps)
        clips.append(clip)

    # Concatenate processed clips
    final_clip = concatenate_videoclips(clips)
    return final_clip

def main_with_buffer(video_filepath, emoji_filepath, query_face_filepath):
    emoji = cv2.imread(emoji_filepath)
    query_faces_embeddings = [extract_embeddings(cv2.imread(x)) for x in query_face_filepath]

    buffer_size = 30  # Set buffer size based on desired batch processing
    processed_video = process_buffered_frames(video_filepath, emoji, query_faces_embeddings, buffer_size)

    out_filepath = os.path.join(os.path.dirname(video_filepath), 'output_' + os.path.basename(video_filepath))
    codec = 'libx264'  # Adjust codec as needed

    # Write the video with the original audio
    processed_video.set_audio(VideoFileClip(video_filepath).audio).write_videofile(out_filepath, codec=codec)

    return out_filepath

if __name__ == '__main__':
    video_path = 'trial_video.mov'
    emoji_path = './smiling_emoji.png'
    query_face_filepath = 'query_face.jpeg'
    out_fp = main_with_buffer(video_path, emoji_path, [query_face_filepath])
    print('Video saved at:', out_fp)
```

### Key Points:
- **Buffer Size**: Choose an appropriate buffer size based on your system's memory capacity and desired processing speed.
- **Batch Processing**: Process frames in batches (buffers) and concatenate results.
- **Concatenate Clips**: Use `concatenate_videoclips` from `MoviePy` to stitch processed clips back together into a single video.

This approach manages frame processing in manageable chunks and can lead to performance gains, especially for resource-intensive operations.



'''

