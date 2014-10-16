function [PV, P, V] = calParam(data, hidActP)
%%
PV = convolutionPV(data, hidActP);

sizeV = size(data, 1) * size(data, 2);
sizeH = size(hidActP, 1) * size(hidActP, 2);
nTimes = size(data, 4) * sizeH; % Fixed me
PV = PV./nTimes;

P = squeeze(sum(sum(sum(hidActP, 1), 2), 3));
P = P./sizeH;

V = squeeze(sum(sum(sum(data, 1), 2), 3));
V =V./sizeV;
end

