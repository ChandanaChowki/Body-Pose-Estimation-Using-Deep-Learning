% Import the Network
clc
close all
%clear all


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
videoFile = 'jumpingjack.mp4';
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

   

    %figure;
    allInterpolated = cell(1, numel(allBodyParts));
    allValues = cell(1, numel(allBodyParts));
    for partIdx = 1:numel(allBodyParts)
        % Extract xy-values of the current body part from all frames
        partXYValues = cellfun(@(framePoses) squeeze(framePoses(:, allBodyParts(partIdx), :)), allPoses, 'UniformOutput', false);

        % Combine xy-values from all frames into a single vector
        allPartXYValues = cat(2, partXYValues{:});
        %Interpolate missing values using interpolation
        interpolatedValues = fillmissing(allPartXYValues, 'linear', 2);

        %store interpolated coordinates for the current body part
        allInterpolated{partIdx} = round(interpolatedValues);
    end

    % Combine interpolated values for all body parts into a single matrix
    allValues = cat(1, allInterpolated{:});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Create a copy of allPoses to store the interpolated poses
    allInterpolatedPoses = allPoses;

    % Replace NaN values in the original poses with the interpolated values
    for frameIdx = 1:numel(allPoses)
        framePoses = zeros(1, 18, 2);
        for partIdx = 1:numel(allBodyParts)
            interpolatedValues = allInterpolated{partIdx};
            partValues =  interpolatedValues(:, frameIdx);

            % Reshape and assign the values to the corresponding body part in the frame poses
            framePoses(1, allBodyParts(partIdx), :) = reshape(partValues, [1, 1, 2]);
        end
        % Store reconstructed poses for this frame
        allInterpolatedPoses{frameIdx} = framePoses;
    end

    % Render Body Poses on the current frame
    renderBodyPoses(im, framePoses, size(heatmaps, 1), size(heatmaps, 2), params);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for frameIdx1 = 1:numel(allPoses)
    framePoses1 = zeros(1,18,2);
    for partIdx1 = 1:numel(bodyPartsOfInterest)
        interpolatedValues1 = allInterpolated{partIdx1};
        partValues1 = interpolatedValues1(:, frameIdx1);
        framePoses1(1, bodyPartsOfInterest(partIdx1), :) = reshape(partValues1, [1,1,2]);
    end
    interestedBodyPoses{frameIdx1} = framePoses1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

framesrepeated = findRepeatedFrames(allInterpolatedPoses);
repeatedFrameIndices = find(framesrepeated);
%disp("indices of repeated frames:");
%disp(repeatedFrameIndices);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

