function [hidActP, poolingOutput, hidState] = pooling(hidInput, poolingScale, poolingType)
%% This file need to be rewriten

hidActP = exp(hidInput); % Here fixed me

scaleSquare = poolingScale^2;
if strcmp(poolingType, 'stochastic')
    tmp = zeros(scaleSquare+1, size(hidActP,1)*size(hidActP,2)*size(hidActP,3)*size(hidActP,4)/scaleSquare);
    tmp(end, :) = 1;
    for c = 1 : poolingScale
        for r = 1 : poolingScale
            k = hidActP(r:poolingScale:end, c:poolingScale:end, :, :);
            tmp((c-1) * poolingScale + r,:) = k(:);
        end
    end
    
    tmp = tmp';
    sumP = sum(tmp, 2);
    tmp = tmp./repmat(sumP, [1, size(tmp, 2)]);
    
    cumP = cumsum(tmp, 2);
    u = rand(size(tmp, 1), 1);
    m = cumP > repmat(u, [1, size(tmp, 2)]);
    index = diff(m, 1, 2);
    S = zeros(size(tmp));
    S(:,1) = 1 - sum(index,2);
    S(:,2:end) = index;
    
    S = S';
    tmp = tmp';
    
    hidState = zeros(size(hidInput));
    
    [row, col, nCase, nFeatureMap] = size(hidInput);
    for c = 1 : poolingScale
        for r = 1 : poolingScale
            hidState(r:poolingScale:end, c:poolingScale:end, :, :) = ...
                reshape(S(poolingScale*(c-1) + r,:), [row/poolingScale, col/poolingScale, nCase, nFeatureMap]);
            hidActP(r:poolingScale:end, c:poolingScale:end, :, :) = ...
                reshape(tmp(poolingScale*(c-1) + r,:), [row/poolingScale, col/poolingScale, nCase, nFeatureMap]);
        end
    end
    
    poolingOutput = zeros(row/poolingScale, col/poolingScale, nCase, nFeatureMap);
    
    
end

end

