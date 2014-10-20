function [data] = trimDataForPooling(data, kernelSize, poolingScale)
%%
if mod(size(data,1)-kernelSize+1, poolingScale)~=0
    n = mod(size(data,1)-kernelSize+1, poolingScale);
    data(1:floor(n/2), : ,:, :) = [];
    data(end-ceil(n/2)+1:end, : ,:, :) = [];
end
if mod(size(data,2)-kernelSize+1, poolingScale)~=0
    n = mod(size(data,2)-kernelSize+1, poolingScale);
    data(:, 1:floor(n/2), :, :) = [];
    data(:, end-ceil(n/2)+1:end, :, :) = [];
end

end

