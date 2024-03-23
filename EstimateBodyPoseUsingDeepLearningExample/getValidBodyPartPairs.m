% For all possible candidates of pairs between body parts A and body parts
% B (bipartite graph), find valid pairs that connect body parts A to body
% parts B such that no two edges share an endpoint and part the affinity
% score is maximized 
% INPUT
%   candidates: matrix of size [numCandidates * 4 (x1,y1,x2,y2)]
%       sorted in descending order based on score
% OUTPUT
%   pairs: matrix of size [numValidPairs * 4(x1,y1,x2,y2)]
function [pairs] = getValidBodyPartPairs(candidates) 
    pairs = zeros(0,4);
    
    for i = 1:size(candidates,1)
        candidate = candidates(i,:); %select candidate
        candidate = squeeze(candidate);
        
        pairAlreadyFound = any(ismember(candidate(1:2),pairs(:,1:2),'rows'),'all') || ... 
                           any(ismember(candidate(3:4),pairs(:,3:4),'rows'),'all');
                       
        if ~pairAlreadyFound
           pairs(end+1,:) = candidate; 
        end
        
    end 
end