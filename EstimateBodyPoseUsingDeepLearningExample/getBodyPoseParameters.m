% Return parameters and constants for OpenPose pose estimation algorithm
function params = getBodyPoseParameters

% Number of body parts in pose
params.NUM_BODY_PARTS = numel(enumeration('BodyParts'));

% Matrix describing how body parts are connected to each other
params.PAIRS = [
    [BodyParts.Neck, BodyParts.LeftShoulder];
    [BodyParts.Neck, BodyParts.RightShoulder];
    [BodyParts.LeftShoulder, BodyParts.LeftElbow];
    [BodyParts.LeftElbow, BodyParts.LeftHand];
    [BodyParts.RightShoulder, BodyParts.RightElbow];
    [BodyParts.RightElbow, BodyParts.RightHand];
    [BodyParts.Neck, BodyParts.LeftHip];
    [BodyParts.LeftHip, BodyParts.LeftKnee];
    [BodyParts.LeftKnee, BodyParts.LeftFoot];
    [BodyParts.Neck, BodyParts.RightHip];
    [BodyParts.RightHip, BodyParts.RightKnee];
    [BodyParts.RightKnee, BodyParts.RightFoot];
    [BodyParts.Neck, BodyParts.Nose];
    [BodyParts.Nose, BodyParts.LeftEye];
    [BodyParts.LeftEye, BodyParts.LeftEar];
    [BodyParts.Nose, BodyParts.RightEye];
    [BodyParts.RightEye, BodyParts.RightEar];
    [BodyParts.LeftShoulder, BodyParts.LeftEar]; % Redundant connection
    [BodyParts.RightShoulder, BodyParts.RightEar]]; % Redundant connection
    
% Indices into the neural network output to obtain the part affinity fields
params.PAF_INDEX = [
    [13 14];
    [21 22];
    [15 16];
    [17 18];
    [23 24];
    [25 26];
    [1 2];
    [3 4];
    [5 6];
    [7 8];
    [9 10];
    [11 12];
    [29 30];
    [31 32];
    [35 36];
    [33 34];
    [37 38];
    [19 20];
    [27 28]];
            
% Number of samples on line when computing the line integral
params.NUM_SAMPLES = 100;

% Pairs to be rendered, excludes redundant connections
params.RENDER_PAIRS = [
    [BodyParts.Neck, BodyParts.LeftShoulder];
    [BodyParts.Neck, BodyParts.RightShoulder];
    [BodyParts.LeftShoulder, BodyParts.LeftElbow];
    [BodyParts.LeftElbow, BodyParts.LeftHand];
    [BodyParts.RightShoulder, BodyParts.RightElbow];
    [BodyParts.RightElbow, BodyParts.RightHand];
    [BodyParts.Neck, BodyParts.LeftHip];
    [BodyParts.LeftHip, BodyParts.LeftKnee];
    [BodyParts.LeftKnee, BodyParts.LeftFoot];
    [BodyParts.Neck, BodyParts.RightHip];
    [BodyParts.RightHip, BodyParts.RightKnee];
    [BodyParts.RightKnee, BodyParts.RightFoot];
    [BodyParts.Neck, BodyParts.Nose];
    [BodyParts.Nose, BodyParts.LeftEye];
    [BodyParts.LeftEye, BodyParts.LeftEar];
    [BodyParts.Nose, BodyParts.RightEye];
    [BodyParts.RightEye, BodyParts.RightEar]];
    
% Colors for the body part pairs for rendering
params.PAIR_COLORS = [
    [255 0 85];
    [255 0 0];
    [255 85 0];
    [255 170 0];
    [255 255 0];
    [170 255 0];
    [85 255 0];
    [0 255 0];
    [0 255 85];
    [0 255 170];
    [0 255 255];
    [0 170 255];
    [0 85 255];
    [0 0 255];
    [255 0 170];
    [170 0 255];
    [255 0 255];
    [85 0 255]];

% Minimum number of parts to be considered a valid pose
params.MIN_PARTS = 5;

% Threshold for part affinity score
params.PAF_THRESH = 0.01;

% Percent of paf scores along line needed to be above the paf threshold
params.PERCENT_ABOVE_THRESH = 0.7;

% Factor to multiply with the heatmap score of a pair to weight it             
params.HEATMAP_SCORE_DISCOUNT = 0.3;

% Heatmap filter threshold
params.FILTER_THRESHOLD = 0.05;

% Filter size for max filter during nonmaximum suppression of heatmap
params.FILTER_WINDOW_SIZE = 5;

end