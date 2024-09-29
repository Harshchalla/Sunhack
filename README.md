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

- Delivering useful results within short time. 

## What's next for Masquerade
- Integrating it into Mental Health apps so that people can open up anonymously.  
