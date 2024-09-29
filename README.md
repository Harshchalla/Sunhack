# The Masquerade Project - Keep it anonymous. Reel is not real. 
Built as a part of sunhacks.io 9/28/24 to 9/29/24 held at Arizona State University

Team members

Aastha Rastogi

Anshika Pandey

Harsh Victor Challa 

Rohan Ahwad

## Inspiration
Sometimes parents want to share their life updates with family but more often than not they don't want to share their kids faces to keep them away from the reel life. Putting young children's picture on social media raises a possibility of identity theft Risk. By 2030, it's estimated that two-thirds of identity fraud cases involving young people will stem from "sharenting". Blurring out faces in videos is not straightforward and it requires expertise in video editing. 

## What it does
Takes a video input from the user and the emoji/mask they want to mask with. The default emoji is smiling face emoji. On submission, the user will see the faces of people that are present in the video, select the person they want to mask, and the processed video is downloaded. Once the masked video is downloaded all the data from the user's video is deleted. 

## How we built it
We built a website hosted on Amazon EC2 instance and used frontend library of Streamlit and Python. The backend code is using Google's Mediapipe Face Mesh Landmark detection, OpenCv and Python.
  
## Challenges we ran into
- We faced a challenge where our model was considering a collar button of a person's face in the video as a face, and an emoji was appearing there itself. So we figured out that it was the threshold that we had to tweak for not considering coordinates of the button significant. 

- We tried extending our project such that the video auto suggests emoji's or mask, based on the person's face. For example, if the person is wearing shades, they would have the 😎 emoji, or if the person is blushing they would have 😊 emoji. On some research, we found a way to manually find the significant face landmarks so that we can find the emoji. We decided to discard the idea since it lacked accuracy and would be too time-consuming, which wasn't feasible within the hackathon's time constraints. 

## Accomplishments that we're proud of
- Successfully delivering what we initially thought of, and meeting our requirements. 

- 
## What we learned
- Working under pressure and in a team.

- Learned technologies like EC2, Mediapipe and Streamlit
  
- Delivering useful results within short time. 

## What's next for Masquerade
- Integration to Mental Health Apps: Integrating it into Mental Health apps so that people can open up anonymously. 
  
- Improving Face Detection Accuracy: While the face detection works well, there's always room to make it even more precise, especially in varied lighting conditions or when there are multiple faces. We aim to refine the model further, possibly incorporating additional facial landmarks and improving the handling of edge cases like objects being mistaken for faces.

- Automated Emoji Suggestions: We plan to revisit the idea of automatically suggesting emojis based on facial features, such as sunglasses or expressions (smiling, winking, etc.). By refining this feature and improving its accuracy, we hope to make the tool more fun and engaging for users while maintaining its core privacy functionality.

- Extended Video Editing Features: We could add additional features like blur effects, background removal, or adding captions automatically to make the platform more versatile for users who want more control over their videos.

- Real-time Face Masking for Live Video: A future goal could be implementing real-time face masking for live streaming or video conferencing, where privacy is equally crucial. This would take the face masking capability to a new level, especially for people concerned about privacy in live broadcasts or calls.

- Integration with Social Media Platforms: Making it easier to directly share the processed videos on popular social media platforms could simplify the process for users who want to post updates while maintaining privacy.

- AI-Driven Privacy Analytics: Another step could involve using AI to analyze and highlight potential privacy risks in videos beyond face masking, offering users suggestions for areas of improvement before sharing their content online.
