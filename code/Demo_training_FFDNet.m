
% This is the training demo of FFDNet for denoising hyperspectral noisy 
% color images corr±upted by AWGN.
%
% To run the code, you should install Matconvnet first. Alternatively, you can use the
% function `vl_ffdnet_matlab` to perform denoising without Matconvnet.
%
% "FFDNet: Toward a Fast and Flexible Solution for CNN based Image
% Denoising" 2018/03/23
clc, clear, close all;
format compact;
addpath('utilities');

run /home/alex/matconv/matlab/vl_setupnn

%-------------------------------------------------------------------------
% Configuration
%-------------------------------------------------------------------------
opts.modelName        = 'models'; % model name
opts.learningRate     = [logspace(-4,-4,100),logspace(-4,-4,100)/3,logspace(-4,-4,100)/(3^2),logspace(-4,-4,100)/(3^3),logspace(-4,-4,100)/(3^4)];% you can change the learning rate
opts.batchSize        = 128; % default  
opts.gpus             = [1]; % this code can only support one GPU!
opts.numSubBatches    = 1;
opts.weightDecay      = 0.0005;
opts.expDir           = fullfile('data', opts.modelName);

%-------------------------------------------------------------------------
%  Initialize model
%-------------------------------------------------------------------------

net  = feval(['model_init_',opts.modelName]);


%-------------------------------------------------------------------------
%   Train
%-------------------------------------------------------------------------
tic
[net, info] = model_train(net,  ...
    'expDir', opts.expDir, ...
    'learningRate',opts.learningRate, ...
    'numSubBatches',opts.numSubBatches, ...
    'weightDecay',opts.weightDecay, ...
    'batchSize', opts.batchSize, ...
    'modelname', opts.modelName, ...
    'gpus',opts.gpus)

toc



