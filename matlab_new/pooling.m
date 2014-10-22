function [hidState, hidActP, poolingOutput] = pooling(hidInput, poolingScale)
%%
[row, col, nFeatureMap] = size(hidInput);
rr = row/poolingScale;
cc = col/poolingScale;

hidActP = exp(hidInput);
tmp = zeros(poolingScale^2 + 1, col*row*nFeatureMap/poolingScale^2);
tmp(end, :) = 1;

for c = 1 : poolingScale
    for r = 1 : poolingScale
        temp = hidActP(r:poolingScale:end, c:poolingScale:end, :);
        tmp((c-1) * poolingScale + r, :) = temp(:);
    end
end

[S1 P1] = multrand2(tmp');
S = S1';
P = P1';
clear S1 P1

hidState = zeros(size(hidInput));
hidActP = zeros(size(hidInput));
for c = 1 : poolingScale
    for r = 1 : poolingScale
        hidState(r:poolingScale:end, c:poolingScale:end, :) = reshape(S((c-1)*poolingScale+r, :), [rr, cc, nFeatureMap]);
        hidActP(r:poolingScale:end, c:poolingScale:end, :) = reshape(P((c-1)*poolingScale+r, :), [rr, cc, nFeatureMap]);
    end
end

poolingOutput = zeros(rr, cc, nFeatureMap);

for i = 1 : nFeatureMap
    index = find(hidState(:, :, i) == 1);
    tmp = hidActP(index); 
    
    c = ceil(index/row);
    r = index - (c-1)*row;

    c = ceil(c/poolingScale);
    r = ceil(r/poolingScale);
    index = (c-1)*rr + r;
    poolingOutput(index, i) = tmp;
end
