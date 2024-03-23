% Import the Network
clc
close all
clear all
dataDir = fullfile(tempdir, 'OpenPose');
trainedOpenPoseNet_url = 'https://ssd.mathworks.com/supportfiles/vision/data/human-pose-estimation.zip'
downloadTrainedOpenPoseNet(trainedOpenPoseNet_url,dataDir);
unzip(fullfile(dataDir,'human-pose-estimation.zip'),dataDir);

modelfile = fullfile(dataDir,'human-pose-estimation.onnx');
layers = importONNXLayers(modelfile,"ImportWeights",true);
%Remove the unused output layers.
layers = removeLayers(layers,layers.OutputNames);
net = dlnetwork(layers);

%Predict Heatmaps and PAFs of Test Image
%Read and display a test image.
im = imread("visionteam.jpg");
imshow(im);
% If it's a color image, rearrange the color channels
netInput = im2single(im)-0.5; %rescaling or offsetting the data as necessary
netInput = netInput(:,:,[3 2 1]); %R,G,B to B,G,R
netInput = dlarray(netInput,"SSC");
[heatmaps,pafs] = predict(net,netInput);
heatmaps = extractdata(heatmaps);
montage(rescale(heatmaps),"BackgroundColor","b","BorderSize",3)
idx = 1;
hmap = heatmaps(:,:,idx);
hmap = imresize(hmap,size(im,[1 2]));
imshowpair(hmap,im);
heatmaps = heatmaps(:,:,1:end-1);
pafs = extractdata(pafs);
montage(rescale(pafs),"Size",[19 2],"BackgroundColor","b","BorderSize",3)
idx = 1;
impair = horzcat(im,im);
pafpair = horzcat(pafs(:,:,2*idx-1),pafs(:,:,2*idx));
pafpair = imresize(pafpair,size(impair,[1 2]));
imshowpair(pafpair,impair);

%Identify Poses from Heatmaps and PAFs
params = getBodyPoseParameters;
poses = getBodyPoses(heatmaps,pafs,params);
renderBodyPoses(im,poses,size(heatmaps,1),size(heatmaps,2),params)


