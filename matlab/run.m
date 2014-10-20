addpath('CDBN', 'CDBN/mex', 'CDBN/args');

option;

X = preprocess('image_files');

kernelSize = opt.kernelSize;
featureMap = opt.featureMap;

CRBM(X, kernelSize(1), featureMap(1), 'inputType', opt.inputType, 'maxEpoch', opt.maxEpoch, 'batchSize', opt.batchSize)