function visActP = reconstruction(hidState, W)
%%
kernelSize = sqrt(size(W, 1));
r = size(hidState, 1);
c = size(hidState, 2);
nFeatureMap = size(W, 2);

visActP = zeros(r + kernelSize - 1, c + kernelSize - 1);

for i = 1:nFeatureMap,
    H = reshape(W(:, i), [kernelSize, kernelSize]);
    visActP = visActP + convolution(hidState(:, :, i), H, 'full');
end
return
