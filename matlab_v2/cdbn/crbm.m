classdef crbm
% Convolutional Restricted Bolztman Machine
% ------------------------------------------------------------------------------------
% Supports binary and gaussian inputs.
% Supports multiple feature maps inputs.
%
% This implementation is based on 'Unsupervised Learning of Hierarchical Representations
% with Convolutional Deep Belief Networks' by Honglak Lee.
%
% -------------------------------------------------------------------------------------
% By shaoguangcheng. Email : chengshaoguang1291@126.com
    properties
        className = 'crbm';
        
        % structure parameters
        inputType = 'binary'; % Supports binary and Gaussian inputs
        nFeatureMapVis;       % number of feature map of visual layer. Default as a single crbm.
        nFeatureMapHid = 10;  % number of feature map of hidden layer. Default as 10.
        visSize;              % size of visual feature map
        kernelSize = [7, 7];  % size of filter
        poolingScale = [2, 2];      % size of stride for x and y axis
        
        % model parameters
        W                     % weight of network
        visBias               % bias of visual layer
        hidBias               % bias of hidden layer
        dW                    % the gradient of weight
        dVisBias
        dHidBias
        
        visSample             % sample result for visual layer
        hidSample             % sample result for hidden layer
        initHidSample         % sample result from the first bottom up
        
        visInput              % input for visual layer 
        hidInput              % input for hidden layer
        outputPooling         % output of pooling
        
        % learning parameters
        nEpoch = 20;
        learningRate = 0.01;
        nCD = 1;
        sparsity = 0.02;
        lambda1 = 1;          % coefficient of sparsity term
        lambda2 = 0.05;       % coefficient of weight decay term
        momentum = 0.9;
        isAnneal = true;      % whether to decay the learning rate
        startWeightDecay = 1; % when to start weight decay
        
        % other parameters
        isUseGPU = 0;
        device;
        verbose = 1;          
        displayInterval = 1;
        outputFolder;
        timeConsumed;
    end
    
    methods
        function self = crbm(netStructure)
            % ---------------------------------
            % intialize crbm
            % ---------------------------------
            
            self = self.checkStructure(netStructure);
            
            % parse parameters
            if isfield(netStructure, 'opt')
                opt = netStructure.opt; % opt must be a cell
                fname = fieldnames(self);
                for i = 1 : 2 : numel(opt)
                    if ~isstr(opt{i})
                        error('opt must be a cell array string-value pairs.')
                    elseif sum(strcmp(fname,opt{i}))
                        self.(opt{i})=opt{i+1};
                    end                    
                end
            end
            
            self.visSize(1) = netStructure.dataSize(1);
            self.visSize(2) = netStructure.dataSize(2);
            self.nFeatureMapVis = netStructure.nFeatureMapVis;
            self.inputType = netStructure.inputType;
            
            if isfield(netStructure, 'nFeatureMapHid')
                self.nFeatureMapHid = netStructure.nFeatureMapHid;
            end
            
            if isfield(netStructure,'kernelSize');
                if numel(netStructure.kernelSize) == 1;
                    self.kernelSize = ones(1,2)*self.kernelSize;
                else
                    self.kernelSize = netStructure.kernelSize;
                end
            end
            
            % initialize parameters
            self.W = 0.01*randn(self.kernelSize(1), self.kernelSize(2), self.nFeatureMapVis, self.nFeatureMapHid);
            self.dW = zeros(size(self.W));
            self.visBias = zeros(self.nFeatureMapVis, 1);
            self.dVisBias = zeros(self.nFeatureMapVis, 1);
            self.hidBias = zeros(self.nFeatureMapHid, 1);
            self.dHidBias = zeros(self.nFeatureMapHid, 1);
            self.visInput = zeros(self.visSize(1), self.visSize(2), self.nFeatureMapVis);
            
            hidSize = self.visSize - self.kernelSize + 1;
            self.hidInput = zeros(hidSize(1), hidSize(2), self.nFeatureMapHid);
        end
        
        function self = checkStructure(self, netStructure)
            % ----------------------------------------
            % check the validation of input parameters
            % ----------------------------------------
           if ~isstruct(netStructure) error('netStructure must be a structure'); end
           if ~isfield(netStructure, 'nFeatureMapVis')||~isfield(netStructure, 'dataSize') error('parameters not enough');end;
           if ~isfield(netStructure, 'inputType')
               self.inputType = 'binary';
           end
        end
        
        function self = train(self, data)
            % ----------------------------
            % train crbm model
            %
            % treat a single example as a batch. In another words,
            % batchsize is 1 according to Honglak Lee's thesis
            % ----------------------------
            tic;
            
            if self.isUseGPU
                self.device = gpuDevice;
                data = gpuarray(data);
            end
            
            nBatch = size(data, 3);
            penalty = self.lambda2;
            
            for epoch = 1:self.nEpoch
               err = 0;
               currentSparity = zeros(1, nBatch);
               if self.isAnneal
                  self.learningRate = max(1e-7, self.learningRate/epoch); 
               end
               
               % set weight decay
               if epoch > self.startWeightDecay
                  self.lambda2 = penalty;
               else
                   self.lambda2 = 0;
               end
               
               % For each epoch, all samples are computed
               for i = 1 : nBatch
                   batchData = data(:,:,i,:);
                   
                   [self, dW, dVisBias, dHidBias] = self.calcGradient(batchData);
                   self = self.applyGradient(dW, dVisBias, dHidBias);
                   
                  if self.sparsity
                    self.hidBias = self.hidBias + ...
                    self.learningRate*self.lambda1*(squeeze(self.sparsity-mean(mean(self.initHidSample))));
                  end
                   
                   err = err + self.batchError(batchData);
                   currentSparity(i) = mean(self.initHidSample(:));
               end
               
               if self.verbose & ~mod(epoch, self.displayInterval)
                   fprintf(1, 'epoch %d, reconstruction error %f, current sparsity %f\n', epoch, err, mean(currentSparity(:)));
               end
            end
            
            self.timeConsumed = toc;
            if self.isUseGPU
                self = gather(self);
                reset(self.device); 
            end
        end
        
        function [self, dW, dVisBias, dHidBias] = calcGradient(self, data)
           % --------------------------------
           % calculate dW, dVisBias, dHidBias
           % --------------------------------
           self = self.gibbsSample(data);
           dW = zeros(size(self.W));
           for i = 1 : self.nFeatureMapHid
              for j = 1 : self.nFeatureMapVis
                  dW(:,:,j,i) = conv2(data(:,:,j), self.ff(self.initHidSample(:,:,i)), 'valid') - ...
                      conv2(self.visSample(:,:,j), self.ff(self.hidSample(:,:,i)), 'valid');
              end 
           end
           
           dVisBias = squeeze(sum(sum(data - self.visSample, 1), 2));
           dHidBias = squeeze(sum(sum(self.initHidSample - self.hidSample, 1), 2));
        end
        
        function self = gibbsSample(self, data)
           % -------------------------
           % do gibbs sampling
           % -------------------------
           % inference
           [self, self.initHidSample] = self.inference(data);
           self.hidSample = self.initHidSample;
           for i = 1 : self.nCD
              self = self.reconstruct(self.hidSample);
              [self, self.hidSample] = self.inference(self.visSample);
           end
           
        end
        
        function [self, hidSample] = inference(self, data)
           % ---------------------------
           % bottom-up process
           % ---------------------------
           self.hidInput = zeros(size(self.hidInput));
           for i = 1 : self.nFeatureMapHid
              for j = 1 : self.nFeatureMapVis
                 self.hidInput(:,:,i) = self.hidInput(:,:,i) + conv2(data(:,:,j), self.ff(self.W(:,:,j,i)), 'valid'); 
              end
               self.hidInput(:,:,i) =  self.hidInput(:,:,i) + self.hidBias(i);
           end
           hidSample = exp(self.hidInput)./(1+self.blockSum(exp(self.hidInput)));
        end
        
        function block = blockSum(self, input)
            % ----------------------------------------------
            % hidden activation summation over block (HongLak Lee's article)
            % ----------------------------------------------
            cols = size(input, 1);
            rows = size(input, 2);
            xStride = self.poolingScale(1);
            yStride = self.poolingScale(2);
            block = zeros(size(input));
            for i = 1 : ceil(rows/yStride)
                offsetRow = ((i-1)*yStride+1):(i*yStride);
               for j = 1 : ceil(cols/xStride)
                   offsetCol = ((j-1)*xStride+1):(j*xStride);
                   blockVal = squeeze(sum(sum(input(offsetRow, offsetCol, :))));
                   block(offsetRow, offsetCol, :) = repmat(permute(blockVal, [2,3,1]), numel(offsetRow), numel(offsetCol));
               end
            end
        end
        
        function self = pooling(self, data)
            % -----------------------------------
            % calcualte the output of pooling layer
            % -----------------------------------
            hidInput = zeros(size(self.hidSample));
           for i = 1 : self.nFeatureMapHid
              for j = 1 : self.nFeatureMapVis
                 hidInput(:,:,i) = self.hidInput(:,:,i) + conv2(data(:,:,j), self.ff(self.W(:,:,j,i)), 'valid'); 
              end
               hidInput(:,:,i) =  self.hidInput(:,:,i) + self.hidBias(i);
           end
           
           hidSample = 1 - (1./(1+self.blockSum(exp(hidInput))));
           xStride = self.poolingScale(1);
           yStride = self.poolingScale(2);
           rows = size(self.hidSample,1);
           cols = size(self.hidSample,2);
           self.outputPooling = hidSample(1:yStride:rows, 1:xStride:cols, :);
        end
        
        function self = reconstruct(self, hidSample)
            % -------------------------
            % top-down process
            % -------------------------
            hidState = double(rand(size(hidSample)) < hidSample);
            self.visInput = zeros(size(self.visInput));
            for i = 1 : self.nFeatureMapVis
                for j = 1 : self.nFeatureMapHid
                    self.visInput(:,:,i) = self.visInput(:,:,i) + conv2(hidState(:,:,j), self.W(:,:,i,j), 'full');
                end
                self.visInput(:,:,i) = self.visInput(:,:,i) + self.visBias(i);
            end
            
            if ~strcmp(self.inputType, 'binary')
                self.visSample = sigmoid(self.visInput);
            else
                self.visSample = self.visInput;
            end  
        end
        
        function self = applyGradient(self, dW, dVisBias, dHidBias)
           % --------------------------
           % apply gradient
           % --------------------------
           [self.W, self.dW] = self.updateParam(self.W, dW, self.dW, self.lambda2);
           [self.visBias, self.dVisBias] = self.updateParam(self.visBias, dVisBias, self.dVisBias, 0);
           [self.hidBias, self.dHidBias] = self.updateParam(self.hidBias, dHidBias, self.dHidBias, 0);
           
        end
        
        function [param, dParam] = updateParam(self, param, inc, dParam, penalty)
           % ----------------------------
           % update parameters
           % ----------------------------
           dParam = self.momentum*dParam + (1 - self.momentum)*inc;
           param = param + self.learningRate*(dParam - penalty*param);
        end
        
        function out = ff(self, in)
            % --------------------------
            % flip array in
            % --------------------------
            out = in(end:-1:1,end:-1:1,:);
        end
        
        function err = batchError(self, data)
            err = (data - self.visSample).^2;
            err = sum(err(:));
        end
        
    end
    
    
end
