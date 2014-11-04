% here we test crbm 
if ispc
    addpath('..\', '..\data', '..\util');
    dataPath = '..\data\mnistSmall';
end

if isunix
    addpath('../', '../data', '../util');
    dataPath = '../data/mnistSmall';
end


load(dataPath);
trainData = reshape(trainData', [28,28,10000]);
trainData = trainData(:,:,1:20);
visSize = [28,28,1];

arch = struct('dataSize', visSize, ...
        'nFeatureMapVis', 1, ...
		'nFeatureMapHid', 10, ...
        'kernelSize', [7 7], ...
        'stride', [2 2], ...
        'inputType', 'binary');

arch.opt = {'nEpoch', 500, ...
			 'learningRate', .05, ...
			 'displayInterval',10, ...
			 'sparsity', .002, ...
			 'lambda1', 5, ...
             'isUseGPU', 0};
         
m = crbm(arch);
m = m.train(trainData);

m = m.pooling(trainData(:,:,1));
m = m.inference(trainData(:,:,1));

figure;
[r,c,n]=size(m.W);
W = reshape(m.W,r*c,n);
subplot(141);
visWeights(W,1);
title('Learned Filters');

subplot(142);
imagesc(trainData(:,:,1)); colormap gray; axis image; axis off
title(sprintf('Sample Data\nPoint'));

subplot(143);
[r,c,n]=size(m.hidSample);
hidSample = reshape(m.hidSample,r*c,n);
visWeights(hidSample);
title('Feature Maps')


subplot(144);
[r,c,n] = size(m.outputPooling);
visWeights(reshape(m.outputPooling,r*c,n)); colormap gray
title(sprintf('Pooling Layer\nExpectations'))
drawnow
