% Get the affinity score of each candidate pair
% INPUT
%   candidates: matrix of size [numCandidates * 4 (x1,y1,x2,y2)]
%   affinityField: matrix of size [fieldHeight * fieldWidth * 2(x and y components of vector field)]
%   params: struct containing constants
% OUTPUT
%   scores: matrix of size [numCandidates * 1 (affinity score)]
function [scores] = getBodyPartAffinityScores(candidates, affinityField, params)
    numCandidates = size(candidates,1);
    scores = zeros(numCandidates, 1);
    
    for i = 1:numCandidates
       candidate = candidates(i,:);
       candidate = squeeze(candidate);
       % Calculate the score as the line integral of the line segment
       % through the vector field The lineIntegral helper function is
       % included in this file.
       scores(i) = lineIntegral(candidate, affinityField, params);
    end    
end

% Calculate the line integral along a line segment between the body parts
% in the body part pair. The vector field defines the affinity between body
% parts with vectors pointing in the direction from one body part to
% another.(that part wasn't worded well) 
% INPUT
%   pair: matrix of size [1 * 4 (x1,y1,x2,y2)]
%   affinityField: matrix of size [fieldHeight * fieldWidth * 2(x and y components of vector field)]
%   params: struct containing constants
% OUTPUT
%   score: scalar representing line integral of the line segment through
%       the vector field, i.e a measure of how much the direction from body
%       part A to body part B "agrees" with the directions of the vectors
%       in the vector field
function [score] = lineIntegral(pair,affinityField,params)
    x1 = pair(1);
    y1 = pair(2);
    x2 = pair(3);
    y2 = pair(4);
    
    v = [x2 y2] - [x1 y1]; % get vector that extends from body part A (x1,y1) to body part B (x2,y2)
    v = v ./ norm(v); % divide vector by it's magnitude to get the unit direction vector
    
    % Sample points along line from A to B, get dot products of direction
    % vector and the vector field and sum to get score
    
    X = floor(linspace(x1,x2,params.NUM_SAMPLES));
    Y = floor(linspace(y1,y2,params.NUM_SAMPLES));
    
    % get linear indices for PAF
    xIndex = repelem(1,params.NUM_SAMPLES);
    xIndices = sub2ind(size(affinityField), Y, X, xIndex);
    yIndex = repelem(2,params.NUM_SAMPLES);
    yIndices = sub2ind(size(affinityField), Y, X, yIndex);
    pafVecsX = affinityField(xIndices);
    pafVecsY = affinityField(yIndices);
    pafVecs = [pafVecsX' pafVecsY'];
    scores = dot(repmat(v, params.NUM_SAMPLES, 1), pafVecs, 2);
    score = mean(scores);
    scoresAboveThresh = scores > params.PAF_THRESH;    
    % score is only valid if certain percent of scores are above threshold
    if sum(scoresAboveThresh) < params.PERCENT_ABOVE_THRESH * params.NUM_SAMPLES
       score = -1; 
    end
end