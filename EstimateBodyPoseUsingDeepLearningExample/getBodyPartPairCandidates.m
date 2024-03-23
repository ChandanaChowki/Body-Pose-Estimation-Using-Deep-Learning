% Generate all possible pairs between the body parts of type A and the body
% parts of type B
% INPUT
%   bodyPartsA: matrix of size [numDetectionsOfA * 2(x,y)]
%   bodyPartsB: matrix of size [numDetectionsOfB * 2(x,y)]
%   scoresA: matrix of size [numDetectionsOfB * 1(detectionScore)]
%   scoresB: matrix of size [numDetectionsOfB * 1(detectionScore)]
% OUTPUT
%   pairCandidates: matrix of size [numCandidates * 4 (x1,y1,x2,y2)]
%   scores: matrix of size [numCandidates * 1(score)]
function [pairCandidates,scores] = getBodyPartPairCandidates(bodyPartsA,bodyPartsB, ...
    scoresA,scoresB)

    % use 'combvec' to find all combinations
    pairCandidates = combvec(bodyPartsA', bodyPartsB');
    pairCandidates = pairCandidates';
    
    scores = combvec(scoresA', scoresB');
    scores = scores';
    scores = sum(scores,2);
end