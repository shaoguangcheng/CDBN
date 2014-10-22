addpath('args/', 'mex/');

option;

dataPath = opt.dataPath;
kernelSize = opt.kernelSize;
featureMap = opt.featureMap;

X = preprocess(dataPath);

CRBML1(X, kernelSize(1), featureMap(1), 'maxEpoch', opt.maxEpoch, 'batchSize', opt.batchSize)


