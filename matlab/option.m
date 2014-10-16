%% do some settings
opt = struct;

%% CDBN settings
opt.inputType  = 'gaussian';
opt.kernelSize =  [2,2];
opt.featureMap = [3,2];
opt.poolingType = 'stochastic';
opt.poolingScale = [2,4];

opt.sigma = 2;
opt.sparsity = 0.002;
opt.lambda1 = 0.01; % The coefficient of weight decay term
opt.lambda2 = 5.0; % The coefficient of sparse term
opt.alpha = 0.01; % learning rate
opt.maxEpoch = 1;
opt.batchSize = 2;
opt.nCD = 1;
opt.biasMode = 'simple';