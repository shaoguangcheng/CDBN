function [hidInput] = inference(data, W, hidBias, stdGaussian)
%%
kernelSize = sqrt(size(W, 1));
nFeatureMapHid = size(W, 2);

hidActP = zeros(size(data, 1) - kernelSize + 1, size(data, 2) - kernelSize + 1, nFeatureMapHid);
hidInput = zeros(size(data, 1) - kernelSize + 1, size(data, 2) - kernelSize + 1, nFeatureMapHid);

W = reshape(W(end:-1:1, :), [kernelSize, kernelSize, nFeatureMapHid]);
hidInput = convolution(data, W, 'valid');

for i = 1 : nFeatureMapHid
    hidInput(:, :, i) = 1/(stdGaussian^2).*(hidInput(:, :, i) + hidBias(i));
    hidActP(:, :, i) = 1./(1 + exp(-hidInput(:, :, i)));
end

return
