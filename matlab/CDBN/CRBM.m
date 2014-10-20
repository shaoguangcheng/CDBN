function [model] = CRBM(X, kernelSize, nFeatureMapHid, varargin)
%%

%% process options
args = prepareArgs(varargin);
[   inputType   ...
    sigma       ...
    poolingScale ...
    poolingType ...
    sparsity    ...
    lambda1     ...
    lambda2     ...
    alpha       ...
    maxEpoch    ...
    batchSize   ...
    nCD         ...
    biasMode    ...
    verbose] = processOptions(args  , ...
    'inputType'  ,  'Bernoulli' , ...
    'sigma'      ,  0.15           , ...
    'poolingScale', 4      , ...
    'poolingType', 'stochastic' , ...
    'sparsity'   ,  0.002       , ...
    'lambda1'    ,  0.01        , ...
    'lambda2'    ,  0.05           , ...
    'alpha'      ,  0.01        , ...
    'maxEpoch'   ,  20          , ...
    'batchSize'  ,  2           , ...
    'nCD'        ,  1           , ...
    'biasMode'   ,  'simple'    , ...
    'verbose'    ,  true);

X = trimDataForPooling(X, kernelSize, poolingScale);

[row, col, nCase, nFeatureMapVis] = size(X);

%% shuffle data
nBatch = ceil(nCase/batchSize);

groups = randperm(nCase);
X = X(:,:,groups,:);

%% initialize parameters
W = 0.01*randn(kernelSize, kernelSize, nFeatureMapHid);
biasV = zeros(nFeatureMapVis, 1);
biasH = -0.1*ones(nFeatureMapHid, 1);

WInc = zeros(kernelSize, kernelSize, nFeatureMapHid);
biasVInc = zeros(nFeatureMapVis, 1);
biasHInc = zeros(nFeatureMapHid, 1);

dW = zeros(kernelSize, kernelSize, nFeatureMapHid);
dBiasV = zeros(nFeatureMapVis, 1);
dBiasH = zeros(nFeatureMapHid, 1);

visActP = zeros(row, col, batchSize, nFeatureMapVis);
hidActP = zeros(row, col, batchSize, nFeatureMapHid);
hidState = zeros(row, col, batchSize, nFeatureMapHid);

%% start 
for epoch = 1 : maxEpoch 
    error = 0;   
    currentSparsity = zeros(1,nBatch);
    for batch = 1 : nBatch
        if batch *batchSize > nCase
            data = X(:,:,(batch-1)*batchSize+1:nCase,:);
        else
            data = X(:,:,(batch-1)*batchSize+1:batch*batchSize,:);
        end
        
        hidInput = inference(data, W, biasH, inputType, sigma);
        [hidActP, poolingOutput, hidState] = pooling(hidInput, poolingScale, poolingType);
        
        [PV1, P1, V1] = calParam(data, hidActP);
        
        for i = 1 : nCD
           visActP = reconstruct(hidState, W, biasV, inputType);
           hidInput = inference(visActP, W, biasH, inputType, sigma);
           [hidActP, poolingOutput, hidState] = pooling(hidInput, poolingScale, poolingType);
        end
        
        [PV2, P2, V2] = calParam(visActP, hidActP);
        
        if strcmp(biasMode, 'simple')
            dBiasH = squeeze(mean(mean(mean(hidActP, 1), 2), 3)) - sparsity;
        end
        
        %%
        
        dW = (PV1 - PV2)/batchSize - lambda1 * W + lambda2 * dW;
        dBiasV = (V1 - V2)/batchSize + lambda2 * dBiasV;
        dBiasH = (P1 - P2)/batchSize + lambda2 * dBiasH;
        
        if epoch < 5
            momentum = 0.5;
        else
            momentum = 0.9;
        end
        
        
        WInc = momentum * WInc + alpha * dW;
        biasVInc = momentum * biasVInc + alpha * dBiasV;
        biasHInc = momentum * biasHInc + alpha * dBiasH;
        
        W = W + WInc;
        biasV = biasV + biasVInc;
        biasH = biasH + biasHInc;
        
        error = error + sum((data(:) - visActP(:)).^2);
        currentSparsity(i) = mean(hidActP(:));
    end
    
    currentSparsity = mean(currentSparsity);
    
    figure(1);
    displayNetwork(reshape(W, [kernelSize*kernelSize, nFeatureMapHid]));
    
    if verbose
        fprintf('Epoch %d, reconstruction error %f, sparsity %f\n', epoch, error, currentSparsity);
    end
    
end

%%
hidInput = inference(X, W, biasH, inputType, sigma);
[hidActP, poolingOutput, hidState] = pooling(hidInput, poolingScale, poolingType);

model.type = inputType;
model.top = poolingOutput;
model.W = W;
model.biasV = biasV;
model.biasH = biasH;

end

