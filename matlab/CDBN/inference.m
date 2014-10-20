function [ hidInput ] = inference(data, W, biasH, inputType, varargin)
%%
nFeatureMapHid = size(W, 3);

hidInput = convolutionVH(data, W);

for i = 1 : nFeatureMapHid
   hidInput(:, :, :, i) = hidInput(:, :, :, i) + biasH(i);
end

if strcmp(inputType, 'gaussian')
    sigma = varargin{1};
    hidInput = 1/sigma^2 .* hidInput;    
end

%% whether need sigmoid here. Fixed me
%hidInput = 1/(1+exp(-1*hidInput));
end

