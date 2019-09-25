clear; clc, close all;
addpath(fullfile('utilities'));
format compact;

%% run matconv directory folder
% run /home/matconv/matlab/vl_setupnn
run \Users\alexm\Desktop\matconvnet-1.0-beta25\matlab\vl_setupnn

%% set parameters
global sigmas; % input noise level 

folderTest  = 'testsets';
folderResult= 'results';
imageSets   = {'Washington-crop-test'};
showResult  = 0;  %% to show the output denoised image
useGPU      = 1; % CPU or GPU. 
pauseTime   = 20;
nch = 25;   %number of channel of the input volume
imageNoiseSigma = 100;  % image noise level (level of the inserted noise)
inputNoiseSigma =100;  % input noise level
num_eval = 1;  %number of running

load(fullfile('data', 'models', 'best_model.mat')); %load model

net.layers = net.layers(1:end-1);
net = vl_simplenn_tidy(net);

if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

vl_simplenn_display(net);

%% define the vector to compute the std (if num_eval>1)
psnr_vector = zeros(1,num_eval);
ssim_vector = zeros(1,num_eval);
msa_vector = zeros(1,num_eval);

PSNR_eval =0;
SSIM_eval =0;
MSA_eval =0;

for eval = 1 : num_eval
mean_PSNR = 0;
mean_SSIM = 0;

for i = 1 : length(imageSets)
   
    %% load testset
    test = load(fullfile(folderTest, imageSets{i}));
    label = test.temp;
    [w,h,depth] = size(label);
    
    %% pre-processing
    K = nch-1; 
    nz = depth + K;
    data = zeros(w,h,nz);
    order_init = (K/2+1):-1:2;
    order_final = (depth-1):-1:(depth-K/2);
    data(:,:,1:K/2) = label(:,:,order_init);
    data(:,:,(K/2 + 1):(end-K/2)) = label;
    data(:,:,(end-K/2+1):end) = label(:,:,order_final);
   
    %% adding simulated AWGN noise 
    sigmas = inputNoiseSigma/255; % see "vl_simplenn.m".
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
    new_img = zeros(nx,ny,depth,'gpuArray');
    
    for z = 1 : depth
        
        input = inputs(:,:,z:z+K);
       
        % perform denoising
        %res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test'); % matconvnet default
        res    = vl_ffdnet_concise(net, input);    % concise version of vl_simplenn for testing 
        output = res(end).x;
        new_img(:,:,z) = output;
    end
    
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
    PSNR=zeros(depth, 1);
    SSIM=zeros(depth, 1);
   
   for band = 1 : depth
   %calculate PSNR, SSIM
        [psnr_cur, ssim_cur, ~] = Cal_PSNRSSIM(new_img(:, :, band), label(:,:,band), 0, 0);
        PSNR(band,1)=psnr_cur;
        SSIM(band,1)=ssim_cur;
   end
   %compute SAM
   [SAM1, SAM2]=SAM(label, new_img);
    
    if showResult
        %input/ Groundtruth / predicted
        %show false color image
        denoised_img = cat(3,im2uint8(new_img(:,:,57)),im2uint8(new_img(:,:,27)),im2uint8(new_img(:,:,17)));
        original_img = cat(3,im2uint8(label(:,:,57)),im2uint8(label(:,:,27)),im2uint8(label(:,:,17)));
        noisy_img = cat(3,im2uint8(inputs(:,:,57)),im2uint8(inputs(:,:,27)),im2uint8(inputs(:,:,17))); 
        figure, imshow(cat(2,noisy_img,original_img,denoised_img));
        %figure, imshow(cat(2, inputs(:,:,15), label(:,:,15), new_img(:,:,15)));
        %figure, imshow(cat(2,im2uint8(input),im2uint8(label),im2uint8(new_img)));
        %title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        %imwrite(im2uint8(output), fullfile(folderResultCur, [nameCur, '_' num2str(imageNoiseSigma,'%02d'),'_' 		  num2str(inputNoiseSigma,'%02d'),'_PSNR_',num2str(PSNRCur*100,'%4.0f'), 'png'] ));
        drawnow;
        pause(pauseTime)
    end
    disp([num2str(mean(PSNR),'%2.4f'),'dB','    ',num2str(mean(SSIM),'%2.4f'),'    MSA: ',num2str(SAM1,'%2.4f')])
    
    PSNR_eval = PSNR_eval + mean(PSNR);
    SSIM_eval = SSIM_eval + mean(SSIM);
    MSA_eval = MSA_eval + SAM1;

end
psnr_vector(eval) = mean(PSNR);
ssim_vector(eval) = mean(SSIM);
msa_vector(eval) = mean(SAM1);

%% show cropped images
denoised_img = cat(3,im2uint8(new_img(:,:,57)),im2uint8(new_img(:,:,27)),im2uint8(new_img(:,:,17)));
noisy_img = cat(3,im2uint8(inputs(:,:,57)),im2uint8(inputs(:,:,27)),im2uint8(inputs(:,:,17))); 
crop_img_bild = noisy_img(121:end-30, 41:90, :);
crop_img_street = noisy_img(41:90, 86:135, :);
    
figure, imshow(crop_img_bild);
figure, imshow(crop_img_street);
    
end
    

std_psnr = std(psnr_vector);
std_ssim = std(ssim_vector);
std_msa = std(msa_vector);
std_time = std(elapsed_time);

PSNR_eval = mean(psnr_vector);
SSIM_eval = mean(ssim_vector);
MSA_eval = mean(msa_vector);
time_eval = mean(elapsed_time);

%% metrics
fileID = fopen('results/Results_wash_evaluation.txt','a');
fprintf(fileID, 'test-set: Washington-----noise level %d \n', imageNoiseSigma);
fprintf(fileID,'PSNR: %2.4f +/- %2.4f \n', [PSNR_eval, std_psnr]);
fprintf(fileID,'SSIM: %2.4f +/- %2.4f \n', [SSIM_eval, std_ssim]);
fprintf(fileID,'MSA: %2.4f +/- %2.4f \n', [MSA_eval, std_msa]);
fclose(fileID);
    







