%% do some settings
opt = struct;

%% data setting
dataPath = '';

%% CDBN settings
opt.inputType  = 'gaussian';
opt.kernelSize =  [10,2];
opt.featureMap = [24,2];
opt.poolingType = 'stochastic';
opt.poolingScale = [2,4];

opt.sigma = 2;
opt.sparsity = 0.002;
opt.lambda1 = 0.01; % The coefficient of weight decay term
opt.lambda2 = 5.0; % The coefficient of sparse term
opt.alpha = 0.01; % learning rate
opt.maxEpoch = 500;
opt.batchSize = 1;
opt.nCD = 1;
opt.biasMode = 'simple';