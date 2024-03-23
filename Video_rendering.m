% Import the Network
clc
close all
clear all


dataDir = fullfile(tempdir, 'OpenPose');
trainedOpenPoseNet_url = 'https://ssd.mathworks.com/supportfiles/vision/data/human-pose-estimation.zip';
downloadTrainedOpenPoseNet(trainedOpenPoseNet_url, dataDir);
unzip(fullfile(dataDir, 'human-pose-estimation.zip'), dataDir);

modelfile = fullfile(dataDir, 'human-pose-estimation.onnx');
layers = importONNXLayers(modelfile, "ImportWeights", true);
% Remove the unused output layers.
layers = removeLayers(layers, layers.OutputNames);
net = dlnetwork(layers);

% Load the video
videoFile = 'groupdance.mp4';
videoObj = VideoReader(videoFile);

% Initialize cell array to store poses for each frame
allPoses = cell(0);

%Define the body parts of interest
bodyPartsOfInterest = [BodyParts.RightHand,BodyParts.LeftHand, BodyParts.RightFoot, BodyParts.LeftFoot];
allBodyParts = 1:18;

% Loop through each frame of the video
while hasFrame(videoObj)
    % Read the current frame
    im = readFrame(videoObj);
    netInput = im2single(im) - 0.5; % rescaling or offsetting the data as necessary
    netInput = netInput(:, :, [3 2 1], :); % R,G,B to B,G,R
    netInput = dlarray(netInput, 'SSC');
    [heatmaps, pafs] = predict(net, netInput);
    heatmaps = extractdata(heatmaps);

    % Identify Poses from Heatmaps and PAFs
    params = getBodyPoseParameters;
    poses = getBodyPoses(heatmaps, pafs, params);

    % Store the poses for the current frame
    allPoses{end+1} = poses;
    renderBodyPoses(im, poses, size(heatmaps, 1), size(heatmaps, 2), params);
end