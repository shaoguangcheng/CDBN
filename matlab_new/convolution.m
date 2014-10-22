function z = convolution(x, y, type)
%%
z = [];
nHidFeatureMap = size(y, 3);
for i=1 : nHidFeatureMap
    z(:, :, i) = conv2(x, y(:,:,i), type);
end

end
