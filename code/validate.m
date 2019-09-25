
function [PSNR_out] = validate(epoch, net, showResult)

format compact;
global sigmas; % input noise level or input noise level map

addpath(fullfile('utilities'));

folderTest  = 'testsets';
folderResult= 'results';
imageSets   = {'Washington-crop-test'};

useGPU      = 1; % CPU or GPU. 
pauseTime   = 0;
nch = 25;

imageNoiseSigma = 100;  % image noise level
inputNoiseSigma = 100;  % input noise level

net.layers = net.layers(1:end-1);
net = vl_simplenn_tidy(net);

if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

for i = 1 : length(imageSets)
    %load testset
    test = load(fullfile(folderTest, imageSets{i}));

    label = test.temp;
 
    %label = im2double(label);
    [w,h,depth] = size(label);
    
    K = nch-1; 
    nz = depth + K;
    data = zeros(w,h,nz);
    order_init = (K/2+1):-1:2;
    order_final = (depth-1):-1:(depth-K/2);
    data(:,:,1:K/2) = label(:,:,order_init);
    data(:,:,(K/2 + 1):(end-K/2)) = label;
    data(:,:,(end-K/2+1):end) = label(:,:,order_final);
    
    % set noise level map (used in vl_simplenn)
    sigmas = inputNoiseSigma/255; % see "vl_simplenn.m".
    
    % add noise
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
    
    [nx,ny,nz] = size(inputs);
    new_img = zeros(nx,ny,depth,'gpuArray');
    
    for z = 1 : depth
        
        input = inputs(:,:,z:z+K);
    
        % perform denoising
        %res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test'); % matconvnet default
        res    = vl_ffdnet_concise(net, input); 
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
    % calculate PSNR, SSIM and save results
        [psnr_cur, ssim_cur, ~] = Cal_PSNRSSIM(new_img(:, :, band), label(:,:,band), 0, 0);
        PSNR(band,1)=psnr_cur;
        SSIM(band,1)=ssim_cur;
    end
    
    disp([num2str(mean(PSNR),'%2.2f'),'dB','    ',num2str(mean(SSIM),'%2.4f')])
  
    fileID = fopen('results/Results_validation.txt','a');
    fprintf(fileID, 'test-set: %s  -  epoch: %d\n', imageSets{i}, epoch);
    fprintf(fileID,'PSNR: %2.2f - SSIM: %2.4f\n', [mean(PSNR), mean(SSIM)]);
    fclose(fileID);   
    
    PSNR_out = mean(PSNR);
end
end
