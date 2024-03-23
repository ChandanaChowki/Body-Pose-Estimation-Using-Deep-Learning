% INPUT
%   heatmaps:  matrix of size [heatmapHeight * heatmapWidth * numBodyParts]
%       one heatmap for each body part showing probability that particular
%       body part is located at a particular location in the image
%   params: struct with constants
% OUTPUT
%   bodyParts: cell array of size [numBodyParts * 1] where each cell is
%       matrix of size [numDetections(variable) * 2(x,y)]
%   detectionScores: cell array of size [numBodyParts * 1] where each cell is
%       matrix of size [numDetections(variable) * 1(detectionScore)]
function [bodyParts,detectionScores] = detectBodyPartsFromHeatmaps(heatmaps,params)
    bodyParts = cell(params.NUM_BODY_PARTS,1);
    detectionScores = cell(params.NUM_BODY_PARTS,1);
    
    for i = 1:params.NUM_BODY_PARTS
        
        %get the heatmap for the current body part type
        h = heatmaps(:,:,i);
        
        %suppress values below threshold
        h(h < params.FILTER_THRESHOLD) = NaN;

        %apply max filter to heatmap
        maxImage = imdilate(h,ones(params.FILTER_WINDOW_SIZE));

        %suppress non-max values
        maxes = (h == maxImage);%maxes occur where the heatmap is equal to the max filter
        [rows,cols] = ind2sub(size(maxes),find(maxes)); %find nonzero indices
        bodyParts{i} = [cols rows];
        detectionScores{i} = h(maxes);            
    end
end