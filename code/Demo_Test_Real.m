clear; clc, close all;
%run /home/alex/matconv/matlab/vl_setupnn
format compact;
global sigmas; % input noise level or input noise level map

addpath(fullfile('utilities'));

folderModel = 'model';
folderTest  = 'real_dataset';
folderResult= 'results';

imageSets   = {'Indian'}
% imageSets   = {'Pavia'}

showResult  = 0;
useGPU      = 1; % CPU or GPU. 
pauseTime   = 0;
nch = 25;

inputNoiseSigma = 50;  % input noise level

%load model
load(fullfile('data', 'models', 'best_model.mat')); %load model

net.layers = net.layers(1:end-1);
net = vl_simplenn_tidy(net);

if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

%vl_simplenn_display(net);
    
for i = 1 : length(imageSets)
    
   
    %load testset
    test = load(fullfile(folderTest, imageSets{i}));
    label = test.img;
    [w,h,depth] = size(label);
    
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

    % tic;
    if useGPU
        inputs = gpuArray(inputs);
    end
    
    [nx,ny,nz] = size(inputs);
    new_img = zeros(nx,ny,depth,'gpuArray');

    for z = 1 : depth
        
        input = inputs(:,:,z:z+K);
        tic
        % perform denoising
        %res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test'); % matconvnet default
        res    = vl_ffdnet_concise(net, input);    % concise version of vl_simplenn 
        output = res(end).x;
        
        new_img(:,:,z) = output;
    end
    

    fileID = fopen('results/Results_elapsed_time_indian.txt','a');
    fprintf(fileID, 'test-set: %s\n', 'test_indian_1band');
    fprintf(fileID,'elapsed_time: %2.4f +/- %2.4f \n', [time_eval, std_time]);
    fclose(fileID);
    
    if mod(w,2)==1
        new_img = new_img(1:end-1,:,:);
        inputs  = inputs(1:end-1,:,:);
    end
    if mod(h,2)==1
        new_img = new_img(:,1:end-1,:);
        inputs  = inputs(:,1:end-1,:);
    end
    
    if useGPU
        new_img = gather(new_img);
        inputs  = gather(inputs);
    end
    inputs = inputs(:,:,K/2+1:end-K/2);
  
    
    if showResult
        %input/ Groundtruth / predicted
        % %show false color image
        denoised_img = im2uint8(new_img(:,:,2));
        original_img = im2uint8(label(:,:,2));
        figure, imshow(cat(2,original_img,denoised_img));
        
%         %indian
%         denoised_img = cat(3,im2uint8(new_img(:,:,2)),im2uint8(new_img(:,:,3)),im2uint8(new_img(:,:,203)));
%         original_img = cat(3,im2uint8(label(:,:,2)),im2uint8(label(:,:,3)),im2uint8(label(:,:,203)));
        
        %pavia
        denoised_img = cat(3,im2uint8(new_img(:,:,97)),im2uint8(new_img(:,:,3)),im2uint8(new_img(:,:,2)));
        original_img = cat(3,im2uint8(label(:,:,97)),im2uint8(label(:,:,3)),im2uint8(label(:,:,2)));
        
        figure, imshow(denoised_img);
        drawnow;
        pause(pauseTime)
    end
%     image_name = strcat('denoised_real/pavia/denoised_pavia_noiselevel_', int2str(inputNoiseSigma));%, int2str(inputNoiseSigma)); 
%     save(image_name,'new_img');
end


fileID = fopen('results/Results_elapsed_time_indian.txt','w');
fprintf(fileID, 'test-set: %s\n', 'test_indian');
fprintf(fileID,'elapsed_time: %2.4f +/- %2.4f \n', [time_eval, std_time]);
fclose(fileID);







