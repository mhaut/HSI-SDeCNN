
function [imdb] = generatepatches(nch)

%-------------------------------------------------------------------------
  % parameter setting
%-------------------------------------------------------------------------
stride     = 20;  % control the number of image patches
patchsize  = 20;

batchSize  = 128; % important for BNorm
count      = 0;   

step1      = 0;
step2      = 0;
sp_stride = 1;

%-------------------------------------------------------------------------
  % patch extraction for the training
%-------------------------------------------------------------------------

%images for the training of the network
train_images = dir(['trainset/*.mat']);

for i = 1 : length(train_images)
    
    train = load(strcat('trainset/', train_images(i).name)); 
 
    tr_data = train.tr_data;
    [w, h, depth] = size(tr_data);
    
    %adding bands 
    K = nch-1; 
    nz = depth + K;
    data = zeros(w,h,nz);
    order_init = (K/2+1):-1:2;
    order_final = (depth-1):-1:(depth-K/2);
    data(:,:,1:K/2) = tr_data(:,:,order_init);
    data(:,:,(K/2 + 1):(end-K/2)) = tr_data;
    data(:,:,(end-K/2+1):end) = tr_data(:,:,order_final);
        
    for j = 1+K/2 : sp_stride : nz-K/2
     
    HR = data(:,:,j-K/2:j+K/2);
        
        [hei,wid,~] = size(HR);
        for x = 1+step1 : stride : (hei-patchsize+1)
            for y = 1+step2 : stride : (wid-patchsize+1)
                count=count+1;
            end
        end
    end
end


numPatches  = ceil(count/batchSize)*batchSize;
diffPatches = numPatches - count;
disp([numPatches,numPatches/batchSize,diffPatches]);

disp('-----------------------------');

count = 0;

HRlab  = zeros(patchsize, patchsize, nch, numPatches,'single');
for i = 1: length(train_images)
    
    train = load(strcat('trainset/', train_images(i).name));
    tr_data = train.tr_data;
    [w, h, depth] = size(tr_data);
    
    nz = depth + K;
    data = zeros(w,h,nz);
    order_init = (K/2+1):-1:2;
    order_final = (depth-1):-1:(depth-K/2);
    data(:,:,1:K/2) = tr_data(:,:,order_init);
    data(:,:,(K/2 + 1):(end-K/2)) = tr_data;
    data(:,:,(end-K/2+1):end) = tr_data(:,:,order_final);
    
    for j = 1+K/2 : sp_stride : nz-K/2
        
        HR  = data(:,:,j-K/2:j+K/2);
        [hei,wid,~] = size(HR);
        for x = 1+step1 : stride : (hei-patchsize+1)
            for y = 1+step2 : stride : (wid-patchsize+1)
                count = count + 1;
                HRlab(:, :, :, count) = HR(x : x+patchsize-1, y : y+patchsize-1,:);
                if count<=diffPatches
                    HRlab(:, :, :, end-count+1)   = HR(x : x+patchsize-1, y : y+patchsize-1, :);
                end
            end
        end
    end
    
end

HRlab_rot  = zeros(patchsize, patchsize, nch, numPatches,'single');
naugment = randi(8,[1,numPatches]);

%data augmentation
for num = 1 : numPatches
    HR_data = HRlab(:,:,:,num);
    HR_rot =data_augmentation(HR_data, naugment(num));
    HRlab_rot(:, :, :, num)   = HR_rot;
end

disp([numPatches*2,(numPatches/batchSize)*2,diffPatches]);
HR_final = cat(4, HRlab, HRlab_rot);

vars = {'HRlab', 'HRlab_rot', 'naugment', 'data', 'HR', 'train', 'tr_data'};
clear(vars{:});

order = randperm(size(HR_final,4));
HR_final = HR_final(:, :, :, order);

imdb.HRlabels = single(HR_final);
imdb.set    = uint8(ones(1,size(imdb.HRlabels,4)));

