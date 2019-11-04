
function [PSNR_out] = validate(epoch, net, showResult, folderResults, nch)

format compact;
global sigmas; % input noise level or input noise level map

%-------------------------------------------------------------------------
% Parameter setting
%-------------------------------------------------------------------------

folderTest  = 'validationset'; %in this case we have used the the test image as validation set
image_name   = 'Washington-crop-test';

useGPU      = 1; % CPU or GPU. For single-threaded (ST) CPU computation, use "matlab -singleCompThread" to start matlab.

imageNoiseSigma = 100;  % image noise level
inputNoiseSigma = 100;  % input noise level

net.layers = net.layers(1:end-1);
net = vl_simplenn_tidy(net);

if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

%load testset
test = load(fullfile(folderTest, image_name));
label = test.temp;
 
label = im2double(label);
[w,h,depth] = size(label);

%-------------------------------------------------------------------------
% Pre-processing
%-------------------------------------------------------------------------

% add bands in order to apply the denoising process to all the bands of
% the considered HSI
K = nch-1; 
nz = depth + K;
data = zeros(w,h,nz);
order_init = (K/2+1):-1:2;
order_final = (depth-1):-1:(depth-K/2);
data(:,:,1:K/2) = label(:,:,order_init);
data(:,:,(K/2 + 1):(end-K/2)) = label;
data(:,:,(end-K/2+1):end) = label(:,:,order_final);
    
% set noise level map 
sigmas = inputNoiseSigma/255;
    
% inserting AWGN noise
randn('seed',0);
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
    
[nx,ny,~] = size(inputs);

%-------------------------------------------------------------------------
% Denoising process
%-------------------------------------------------------------------------
 
output_img = zeros(nx,ny,depth,'gpuArray');
  
for z = 1 : depth
    
    input = inputs(:,:,z:z+K);
    
    %perform denoising one band at time
    %res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test'); % matconvnet default
    res    = vl_net_concise(net, input); %faster
    output = res(end).x;
        
    output_img(:,:,z) = output;
end

if mod(w,2)==1
   output_img = output_img(1:end-1,:,:);
   inputs  = inputs(1:end-1,:,:);
end
 
if mod(h,2)==1
   output_img= output_img(:,1:end-1,:);
   inputs  = inputs(:,1:end-1,:);
end
    
if useGPU
   output_img = gather(output_img);
   inputs  = gather(inputs);
end
   
inputs = inputs(:,:,K/2+1:end-K/2);

%-------------------------------------------------------------------------
% Compute statistics and plot results
%-------------------------------------------------------------------------

PSNR=zeros(depth, 1);
SSIM=zeros(depth, 1);
    

for band = 1 : depth
    %calculate PSNR, SSIM and save results
    [psnr_cur, ssim_cur, ~] = Cal_PSNRSSIM(output_img(:, :, band), label(:,:,band), 0, 0);
    PSNR(band,1)=psnr_cur;
    SSIM(band,1)=ssim_cur;
end

if showResult   %show false color image
    %input/ Groundtruth / predicted
    denoised_img = cat(3,im2uint8(output_img(:,:,57)),im2uint8(output_img(:,:,23)),im2uint8(output_img(:,:,17)));
    original_img = cat(3,im2uint8(label(:,:,57)),im2uint8(label(:,:,23)),im2uint8(label(:,:,17)));
    noisy_img = cat(3,im2uint8(inputs(:,:,57)),im2uint8(inputs(:,:,23)),im2uint8(inputs(:,:,17))); 
    figure, imshow(cat(2,noisy_img,original_img,denoised_img));
end
    
disp([num2str(mean(PSNR),'%2.2f'),'dB','    ',num2str(mean(SSIM),'%2.4f')])


if ~exist(folderResults, 'dir'), mkdir(folderResults) ; end

fileID = fopen(fullfile(folderResults,'Results_validation.txt'),'a');
fprintf(fileID, 'test-set: %s  -  epoch: %d\n', image_name, epoch);
fprintf(fileID,'PSNR: %2.2f - SSIM: %2.4f\n', [mean(PSNR), mean(SSIM)]);
fclose(fileID); 

PSNR_out = mean(PSNR);
    

