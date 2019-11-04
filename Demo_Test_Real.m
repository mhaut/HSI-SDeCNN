
clear; clc, close all;

addpath(fullfile('utilities'));
format compact;

% run vl_setupnn in the MatConvNet directory
run /home/alex/matconvnet-1.0-beta25/matlab/vl_setupnn

%-------------------------------------------------------------------------
  % parameter setting
%-------------------------------------------------------------------------

global sigmas; %input noise level

folderTest  = 'testsets/real_datasets';
folderResults= 'Results';
save_result = 1; %1 if you want t save the denoised HSI, 0 otherwise

imageSet   = {'Indian'};
%imageSet   = {'Pavia'};

showResult  = 1;
useGPU      = 1; % CPU or GPU. 
nch = 25;

inputNoiseSigma = 50;  % input noise level

%-------------------------------------------------------------------------
  % load model
%-------------------------------------------------------------------------

load(fullfile('BestModel','best_model')); %load model

net.layers = net.layers(1:end-1);
net = vl_simplenn_tidy(net);

if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

%vl_simplenn_display(net);

%-------------------------------------------------------------------------
  % load target HSI
%-------------------------------------------------------------------------
test = load(fullfile(folderTest, imageSet{1}));
label = test.img;
[w,h,depth] = size(label);

%-------------------------------------------------------------------------
  % pre-processing
%-------------------------------------------------------------------------

K = nch-1; 
nz = depth + K;
data = zeros(w,h,nz);
order_init = (K/2+1):-1:2;
order_final = (depth-1):-1:(depth-K/2);
data(:,:,1:K/2) = label(:,:,order_init);
data(:,:,(K/2 + 1):(end-K/2)) = label;
data(:,:,(end-K/2+1):end) = label(:,:,order_final);
    

sigmas = inputNoiseSigma/255; % see "vl_simplenn.m".
inputs = data;

if mod(w,2)==1
    inputs = cat(1,inputs, inputs(end,:,:)) ;
end

if mod(h,2)==1
    inputs = cat(2,inputs, inputs(:,end,:)) ;
end

if useGPU
    inputs = gpuArray(inputs);
end
    
[nx,ny,nz] = size(inputs);
output_img = zeros(nx,ny,depth,'gpuArray');

%-------------------------------------------------------------------------
  % denoising process
%-------------------------------------------------------------------------

tic
for z = 1 : depth
        
    input = inputs(:,:,z:z+K);

    % perform denoising
    % res = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test'); % matconvnet default
    res = vl_net_concise(net, input);    % concise version of vl_simplenn 
    output = res(end).x;
        
    output_img(:,:,z) = output;
end

toc

if mod(w,2)==1
    output_img = output_img(1:end-1,:,:);
    inputs  = inputs(1:end-1,:,:);
end

if mod(h,2)==1
    output_img = output_img(:,1:end-1,:);
    inputs  = inputs(:,1:end-1,:);
end
    
if useGPU
    output_img = gather(output_img);
    inputs  = gather(inputs);
end

inputs = inputs(:,:,K/2+1:end-K/2);
  
    
if showResult
    % input/ Groundtruth / predicted
    % show false color image
    denoised_img = im2uint8(output_img(:,:,2));
    original_img = im2uint8(label(:,:,2));
    figure, imshow(cat(2,original_img,denoised_img));
    
    % Indian-pines
    if strcmp(imageSet{1}, 'Indian')
        denoised_img = cat(3,im2uint8(output_img(:,:,2)),im2uint8(output_img(:,:,3)),im2uint8(output_img(:,:,203)));
        original_img = cat(3,im2uint8(label(:,:,2)),im2uint8(label(:,:,3)),im2uint8(label(:,:,203)));
        
    % University of Pavia
    elseif strcmp(imageSet{1}, 'Pavia')
        denoised_img = cat(3,im2uint8(output_img(:,:,97)),im2uint8(output_img(:,:,3)),im2uint8(output_img(:,:,2)));
        original_img = cat(3,im2uint8(label(:,:,97)),im2uint8(label(:,:,3)),im2uint8(label(:,:,2)));
    end
    
    figure, imshow(cat(2,original_img,denoised_img));

end

if ~exist(fullfile(folderResults, imageSet{1}), 'dir'), mkdir(fullfile(folderResults, imageSet{1})); end

if save_result   
    image_name = strcat(fullfile(folderResults, imageSet{1}, 'denoised_noiselevel_'), int2str(inputNoiseSigma));
    save(image_name,'output_img');
end
