addpath('CDBN', 'CDBN/mex', 'CDBN/args');

option;

xx = [1,2,3;4,5,6];
xxx(:,:,1) = xx;
xxx(:,:,2) = xx;
xxx(:,:,3) = xx;

X = zeros(2, 3, 3, 3);
X(:,:,:,1) = xxx;
X(:,:,:,2) = xxx;
X(:,:,:,3) = xxx;

kernelSize = opt.kernelSize;
featureMap = opt.featureMap;

CRBM(X, kernelSize(1), featureMap(1), 'inputType', opt.inputType, 'maxEpoch', opt.maxEpoch, 'batchSize', opt.batchSize)