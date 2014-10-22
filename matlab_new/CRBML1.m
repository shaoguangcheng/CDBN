function [model] = CRBML1(X, kernelSize, nFeatureMapHid, varargin)

args = prepareArgs(varargin);
[   patchSize   ...
    sigmaStart  ...
    sigmaStop   ...
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
    'patchSize'  ,  70          , ...
    'sigmaStart' ,  0.2         , ...
    'sigmaStop'  ,  0.1         , ...
    'poolingScale', 2           , ...
    'poolingType', 'stochastic' , ...
    'sparsity'   ,  0.002       , ...
    'lambda1'    ,  0.01        , ...
    'lambda2'    ,  5           , ...
    'alpha'      ,  0.01        , ...
    'maxEpoch'   ,  20          , ...
    'batchSize'  ,  2           , ...
    'nCD'        ,  1           , ...
    'biasMode'   ,  'simple'    , ...
    'verbose'    ,  true);


if mod(kernelSize,2)~=0
    disp('ws must be even number');
end

W = 0.01*randn(kernelSize^2, nFeatureMapHid);
visBias = 0;
hidBias = -0.1*ones(nFeatureMapHid,1);
WInc=0;
visBiasInc=0;
hidBiasInc=0;

stdGaussian = sigmaStart;
initialmomentum  = 0.5;
finalmomentum    = 0.9;
nCase = size(X, 3);

for epoch=1:maxEpoch
    error = zeros(1,nCase*batchSize);
    currentSparsity = zeros(1,nCase*batchSize);

   indexData = randperm(size(X,3));
    for i = 1:nCase
        index = indexData(i);
        data = X(:,:,index);
        [rows, cols] = size(data);

        for batch=1:batchSize            
            rowIndex = ceil(rand*(rows-2*kernelSize-patchSize))+kernelSize + [1:patchSize];
            colIndex = ceil(rand*(cols-2*kernelSize-patchSize))+kernelSize + [1:patchSize];
            dataBatch = data(rowIndex, colIndex);
            dataBatch = dataBatch - mean(dataBatch(:));
            dataBatch = trimDataForPooling(dataBatch, kernelSize, poolingScale);
            
            if rand()>0.5,
                dataBatch = fliplr(dataBatch);
            end
            
            hidInput = inference(dataBatch, W, hidBias, stdGaussian);
            [hidState hidActP] = pooling(hidInput, poolingScale);            
            tmp = hidActP;
            
            [PV1, P1, V1] = calParam(dataBatch, hidActP, kernelSize);
            
            for j = 1 : nCD
                visActP = reconstruction(hidState, W);
                hidInput = inference(visActP, W, hidBias, stdGaussian);               
                [hidState hidActP] = pooling(hidInput, poolingScale);
            end
            
            [PV2, P2, V2] = calParam(visActP, hidActP, kernelSize);
                       
            if strcmp(biasMode, 'none')
                dHidBias = 0;
                dVisBias = 0;
                dW = 0;
            elseif strcmp(biasMode, 'simple')
                dHidBias = squeeze(mean(mean(tmp,1),2))-sparsity;
                dVisBias = 0;
                dW = 0;
            end
            
            error(i) = sum((dataBatch(:)-visActP(:)).^2);
            currentSparsity(i) = mean(tmp(:));

            if epoch < 5,
                momentum = initialmomentum;
            else
                momentum = finalmomentum;
            end
                       
            numCase = size(tmp,1)*size(tmp,2);
            dW = (PV1 - PV2)/numCase - lambda1 * W - lambda2 * dW;
            dHidBias = (P1 - P2)/numCase - lambda2 * dHidBias;
            dVisBias = (V1 - V2)/numCase;
           
            WInc = momentum * WInc + alpha * dW;
            visBiasInc = momentum*visBiasInc + alpha * dVisBias;
            hidBiasInc = momentum*hidBiasInc + alpha * dHidBias;
            
            W = W + WInc;
            visBias = visBias + visBiasInc;
            hidBias = hidBias + hidBiasInc;
        end

        if (stdGaussian > sigmaStop)
            stdGaussian = stdGaussian * 0.99;
        end

    end

    error= mean(error);
    currentSparsity = mean(currentSparsity);

    figure(1);
    displayNetwork(W);

    fprintf('Epoch %d reconstruction error = %f, sparsity = %f\n', epoch, error, currentSparsity);
end

model = {};
model.W = W;
model.visBias = visBias;
model.hidBias = hidBias;
top = inference();


