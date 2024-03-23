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

% If it's the first frame, store the selected poses

selectedPoses{1} = allInterpolatedPoses{1};
selectedPoses{2} = allInterpolatedPoses{5};
selectedPoses{3} = allInterpolatedPoses{10};

%threshold = 3;

% Initialize cell array to store similarity scores for each selected pose
similarityScores = cell(numel(selectedPoses), 1);

% Initialize logical array to keep track of frames assigned to a pose
assignedFrames = false(1, numel(allInterpolatedPoses));
% Initialize assignedPose cell array with false arrays
assignedPose = cell(1, numel(selectedPoses));
numFrames = numel(allInterpolatedPoses);
for i = 1:numel(selectedPoses)
    assignedPose{i} = false(1, numFrames);
end

% Iterate over each selected pose
for i = 1:numel(selectedPoses)
    % Get the current selected pose
    selectedPose = selectedPoses{i};

    % Reshape the pose to concatenate x and y coordinates
    selectedPoseVector = reshape(selectedPose, [], 2);

    % Initialize array to store similarity scores for the current pose
    poseSimilarityScores = zeros(1, numel(allInterpolatedPoses));


    % Compare selected poses with poses in all frames
    for j = 1:numel(allInterpolatedPoses)
        framePose = allInterpolatedPoses{j};
        framePoseVector = reshape(framePose, [], 2);
        jointDistances = sqrt(sum((selectedPoseVector - framePoseVector).^2, 2));
        poseSimilarityScores(j) = mean(jointDistances);
    end
    
    similarityScores{i} = round(poseSimilarityScores, 2);
%     % Consider frames with similarity scores less than 3.5 for this pose
%     assignedFrames = assignedFrames | ((poseSimilarityScores < threshold) & ~any(assignedFrames));
    
    
end

% Assign frames to the nearest selected pose based on similarity scores
for frameIdx = 1:numel(allInterpolatedPoses)
    minDistances = inf(1, numel(selectedPoses));
    
    % Iterate over each selected pose to find the minimum similarity score
    for poseIdx = 1:numel(selectedPoses)
        % Get the similarity scores for the current pose
        scores = similarityScores{poseIdx};
        
        % Find the minimum similarity score for the current frame index
        minDistances(poseIdx) = min(scores(frameIdx));
    end
    
    % Find the index of the nearest selected pose based on the minimum similarity score
    [~, nearestPoseIdx] = min(minDistances);
    
    % Update assigned pose for the current frame
    assignedPose{nearestPoseIdx}(frameIdx) = true;
end

% Display assigned frames for each pose
for poseIdx = 1:numel(selectedPoses)
    fprintf('Assigned frames for Pose %d:\n', poseIdx);
    assignedFrames = find(assignedPose{poseIdx});
    disp(assignedFrames);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize transition matrix

numStates = numel(selectedPoses); % Number of states (poses)
transitionMatrix = zeros(numStates); % Initialize transition matrix

% Iterate over assigned poses
for i = 1:numStates
    % Get the assigned pose for the current frame
    currentPose = assignedPose{i};

    % Iterate over assigned poses for the next frame
    for j = 1:numStates
        % Get the assigned pose for the next frame
        nextPose = assignedPose{j};

        % Count transitions from current pose to next pose
        transitionCount = sum(currentPose(1:end-1) & nextPose(2:end));

        % Update transition count in the transition matrix
        transitionMatrix(i, j) = transitionCount;
    end
end

% Normalize transition matrix to get transition probabilities
transitionMatrix = transitionMatrix ./ sum(transitionMatrix, 2);

figure
imagesc(transitionMatrix);
colorbar;
title('Transition Matrix wrt poses');
xlabel('Current State');
ylabel('Next State');




% Function to determine if the values in frames are repeated in other frames
function repeatedFrames = findRepeatedFrames(allFrames)
    numFrames = numel(allFrames);
    repeatedFrames = false(1, numFrames); % Initialize array to store repeated frame indices
    
    % Iterate through each frame
    for i = 1:numFrames
        currentFrame = allFrames{i}; % Get pose of the current frame
        
        % Iterate through other frames to compare with the current frame
        for j = 1:numFrames
            if i == j
                continue; % Skip comparing with the same frame
            end
            
            otherFrame = allFrames{j}; % Get pose of the other frame
            
            % Check if the current frame's pose is equal to the other frame's pose
            if isequal(currentFrame, otherFrame)
                repeatedFrames(i) = true; % Mark the current frame as repeated
                break; % Exit loop once a match is found
            end
        end
    end
end


