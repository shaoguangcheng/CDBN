function [PV, P, V] = calParam(data, hidActP, kernelSize)
nFeatureMap = size(hidActP, 3);

index1 = size(hidActP,1):-1:1;
index2 = size(hidActP,2):-1:1;

PV = convolution(data, hidActP(index1, index2, :), 'valid');
PV = reshape(PV, [kernelSize^2, nFeatureMap]);

P = squeeze(sum(sum(hidActP,1),2));
V = sum(data(:));

return
