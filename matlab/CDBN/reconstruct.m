function [visActP] = reconstruct(hidState, W, biasV, inputType)
%%
nFeatureMapVis = size(biasV, 1);
visActP = convolutionHV(hidState, W, nFeatureMapVis);

for i = 1 : nFeatureMapVis
   visActP(:,:,:,i) = visActP(:,:,:,i) + biasV(i); 
end

if strcmp(inputType, 'gaussian')
     visActP = visActP + randn(size(visActP));
 else
     %% whether need sigmoid here. fixe me
     visActP = 1/(1+exp(-1*visActP));
end
end

