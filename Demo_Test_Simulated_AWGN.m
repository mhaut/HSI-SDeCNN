
clear; clc, close all;

addpath(fullfile('utilities'));
format compact;

% run vl_setupnn in the MatConvNet directory
run /home/alex/matconvnet-1.0-beta25/matlab/vl_setupnn

%-------------------------------------------------------------------------
  % parameter setting
%-------------------------------------------------------------------------

global sigmas; % input noise level 

folderTest  = 'testsets';
folderResults = 'Results';
imageSets   = {'Washington-crop-test'};
showResult  = 0;  % set to 1 to show the output denoised image
useGPU      = 1; % CPU or GPU. 
nch = 25;   %number of channel of the input volume
imageNoiseSigma = 100;  % image noise level (level of the inserted noise)
inputNoiseSigma =100;  % input noise level
num_eval = 1;  %number of running

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

% define the vectors to compute the std (if num_eval>1)
psnr_vector = zeros(1,num_eval);
ssim_vector = zeros(1,num_eval);
msa_vector = zeros(1,num_eval);
elapsed_time = zeros(1,num_eval);

for eval = 1 : num_eval

   
%-------------------------------------------------------------------------
  % load target HSI
%-------------------------------------------------------------------------
    test = load(fullfile(folderTest, imageSets{1}));
    label = test.temp;
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
   
%-------------------------------------------------------------------------
  % inserting simulated AWGN noise
%-------------------------------------------------------------------------   
    sigmas = inputNoiseSigma/255;
    noise = imageNoiseSigma/255.*randn(size(data));
    
    inputs = single(data + noise);
    
    
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
    tic
    
%-------------------------------------------------------------------------
  % denoising process
%-------------------------------------------------------------------------  
    
    for z = 1 : depth
        
        input = inputs(:,:,z:z+K);
       
        % perform denoising
        % res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test'); % matconvnet default
        res    = vl_net_concise(net, input);    % concise version of vl_simplenn for testing (faster) 
        output = res(end).x;
        output_img(:,:,z) = output;
    end
  
     elapsed_time(eval) = toc;
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
    
%-------------------------------------------------------------------------
    % Compute statistics and plot results
%-------------------------------------------------------------------------
    inputs = inputs(:,:,K/2+1:end-K/2);
    PSNR=zeros(depth, 1);
    SSIM=zeros(depth, 1);
   
   for band = 1 : depth
        % calculate PSNR, SSIM
        [psnr_cur, ssim_cur, ~] = Cal_PSNRSSIM(output_img(:, :, band), label(:,:,band), 0, 0);
        PSNR(band,1)=psnr_cur;
        SSIM(band,1)=ssim_cur;
   end
   % compute SAM
   [SAM1, SAM2]=SAM(label, output_img);
    
    if showResult
        % input/ Groundtruth / predicted
        % show false color image
        denoised_img = cat(3,im2uint8(output_img(:,:,57)),im2uint8(output_img(:,:,27)),im2uint8(output_img(:,:,17)));
        original_img = cat(3,im2uint8(label(:,:,57)),im2uint8(label(:,:,27)),im2uint8(label(:,:,17)));
        noisy_img = cat(3,im2uint8(inputs(:,:,57)),im2uint8(inputs(:,:,27)),im2uint8(inputs(:,:,17))); 
        figure, imshow(cat(2,noisy_img,original_img,denoised_img));
        
        % show cropped images
        % crop_img_bild = noisy_img(121:end-30, 41:90, :);
        % crop_img_street = noisy_img(41:90, 86:135, :);         
        % crop_img_bild_denoised = noisy_img(121:end-30, 41:90, :);
        % crop_img_street_denoised = noisy_img(41:90, 86:135, :);
                 
        % figure, imshow(crop_img_bild);
        % figure, imshow(crop_img_street);
        % figure, imshow(crop_img_bild_denoised);
        % figure, imshow(crop_img_street_denoised);
    end
    
    disp([num2str(mean(PSNR),'%2.4f'),'dB','    ',num2str(mean(SSIM),'%2.4f'),'    MSA: ',num2str(SAM1,'%2.4f')]);

psnr_vector(eval) = mean(PSNR);
ssim_vector(eval) = mean(SSIM);
msa_vector(eval) = mean(SAM1);
  
end

%-------------------------------------------------------------------------
    % Save results
%-------------------------------------------------------------------------

std_psnr = std(psnr_vector);
std_ssim = std(ssim_vector);
std_msa = std(msa_vector);
std_time = std(elapsed_time);

PSNR_eval = mean(psnr_vector);
SSIM_eval = mean(ssim_vector);
MSA_eval = mean(msa_vector);
time_eval = mean(elapsed_time);

if ~exist(folderResults, 'dir'), mkdir(folderResults) ; end

fileID = fopen(fullfile(folderResults, 'Results_wash_evaluation.txt'),'w');
fprintf(fileID, 'test-set: Washington-----noise level %d \n', imageNoiseSigma);
fprintf(fileID,'PSNR: %2.4f +/- %2.4f \n', [PSNR_eval, std_psnr]);
fprintf(fileID,'SSIM: %2.4f +/- %2.4f \n', [SSIM_eval, std_ssim]);
fprintf(fileID,'MSA: %2.4f +/- %2.4f \n', [MSA_eval, std_msa]);
fprintf(fileID,'elpased-time: %2.4f +/- %2.4f \n', [time_eval, std_time]);
fclose(fileID);
    
