% INPUT
%   heatmaps: matrix of size [heatmapHeight * heatmapWidth * numBodyParts]
%       one heatmap for each body part showing probability that particular
%       body part is located at a particular location in the image, plus
%       one heatmap for the background
%   pafs: matrix of size [pafHeight * pafWidth * (2*numPairs)] containing part
%       affinity fields for each body part pair. Each paf has x component
%       and y component of vector field pointing from one body part to the
%       other
%   params: struct containing constants
% OUTPUT
%   poses: matrix of size [numPeople * 18(aka numBodyParts) * 2(x,y)]
%       records locations of each body part for each person in the image,
%       the location value is [NaN, NaN] if that body part is not found in
%       the image
function [poses] = getBodyPoses(heatmaps,pafs,params)
    
    % Use nonmaximum suppression to extract body part coordinates from heatmaps.
    [bodyParts,detectionScores] = detectBodyPartsFromHeatmaps(heatmaps,params);
    
    % Create poses structure to be built
    poses = [];
    EMPTY_POSE = repmat([NaN NaN],params.NUM_BODY_PARTS,1);
    poses(end+1,:,:) = EMPTY_POSE; %add empty pose so ismember works
    
    % Loop through each pair type
    for p = 1:size(params.PAIRS,1)
        a = params.PAIRS(p,1);
        b = params.PAIRS(p,2);
        bodyPartsA = bodyParts{a};
        bodyPartsB = bodyParts{b};
        scoresA = detectionScores{a};
        scoresB = detectionScores{b};

        % Generate all possible pair candidates.
        [candidates,heatmapScores] = getBodyPartPairCandidates(bodyPartsA, ...
            bodyPartsB,scoresA,scoresB);
        if isempty(candidates)
           continue 
        end
        
        % Get affinity field for this pair
        px = params.PAF_INDEX(p,1);
        py = params.PAF_INDEX(p,2);
        affinityFieldX = pafs(:,:,px);
        affinityFieldY = pafs(:,:,py);
        affinityField = cat(3,affinityFieldX,affinityFieldY);
        
        % Calculate the scores and sort candidates accordingly
        scores = getBodyPartAffinityScores(candidates,affinityField,params);
        
        % Add PAF scores to heatmap scores, discounting the heatmap scores
        scores = scores + (params.HEATMAP_SCORE_DISCOUNT*heatmapScores);
        candidates = [scores candidates];
        candidates = sortrows(candidates,'descend');
        
        % Only keep candidates with scores greater than 0
        candidates = candidates(candidates(:,1)>0,:);
        candidates = candidates(:,2:5);%remove scores after sorting
        pairs = getValidBodyPartPairs(candidates); 
        
        % Add valid pairs to the poses     
        poses = addBodyPartPairsToBodyPoses(pairs,poses,a,b);
    end
    
    % Remove the dummy pose
    poses = poses(2:end,:,:);
    
    % Filter out invalid poses (poses with < x body parts)
    poses = filterBodyPoses(poses, params);
end