# Body-Pose-Estimation-Using-Deep-Learning

OpenPose algorithm and a pretrained network identifies the location of people in an image and the orientation of their body parts. When multiple people are present in a scene, pose estimation can be more difficult because of occlusion, body contact, and proximity of similar body parts.

There are two strategies to estimating body pose. A top-down strategy first identifies individual people using object detection and then estimates the pose of each person. A bottom-up strategy first identifies body parts in an image, such as noses and left elbows, and then assembles individuals based on likely pairings of body parts. The bottom-up strategy is more robust to occlusion and body contact, but the strategy is more difficult to implement. OpenPose is a multi-person human pose estimation algorithm that uses a bottom-up strategy.

To identify body parts in an image, OpenPose uses a pretrained neural network that predicts heatmaps and part affinity fields (PAFs) for body parts in an input image. Each heatmap shows the probability that a particular type of body part is located at each pixel in the image. The PAFs are vector fields that indicate whether two body parts are connected. For each defined type of body part pairing, such as neck to left shoulder, there are two PAFs that show the x- and y-component of the vector field between instances of the body parts.

To assemble body parts into individual people, the OpenPose algorithm performs a series of post-processing operations. The first operation identifies and localized body parts using the heatmaps returned by the network. Subsequent operations identify actual connections between body parts, resulting in the individual poses. For more details about the algorithm, see Identify Poses from Heatmaps and PAFs.

The goal of my project is to render bodyposes to a video using the algorithm, define poses by analysing the data and find the transition probability from one pose to another. I have interpolated the NaN values, then rendered body poses and considered a HMM model for a transition matrix.

Prerequisites is only having latest MATLAB. Pretrained network can be imported from an ONNX file which is included in the code. OpenPose has also represented the first real-time multi-person system to jointly detect human body, hand, facial, and foot keypoints (in total 135 keypoints) on single images. But for the sake of the project I have only used saved videos. If you want to work on real-time system and other platforms you can refer at https://github.com/CMU-Perceptual-Computing-Lab/openpose.git .

Save the video files that you want to render body poses at the location where the pretrained network is imported. I have interpolated Nan values and rendered poses for the videos that has only one person, for multiple persons add another loop where you can iterate the code for each person present in the video.
