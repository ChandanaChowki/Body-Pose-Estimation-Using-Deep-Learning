% Remove invalid poses
function [poses] = filterBodyPoses(poses,params)
    function [count] = countBodyParts(pose)
       count = 0;       
       for i = 1:size(pose,1)
           if ~isequaln(pose(i,:),[NaN,NaN])
               count = count + 1;
           end
       end
    end

    %vectorize?
    invalidPoses = [];
    for p = 1:size(poses,1)
        if countBodyParts(squeeze(poses(p,:,:))) < params.MIN_PARTS
            invalidPoses(end+1) = p;
        end
    end
    poses(invalidPoses,:,:) = [];
end