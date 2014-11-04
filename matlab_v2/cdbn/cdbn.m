classdef cdbn
    % Convolution Deep Belif Network (CDBN)
    % -------------------------------------------
    % This implementation is based on 'Unsupervised Learning of Hierarchical Representations
    % with Convolutional Deep Belief Networks' by Honglak Lee. 
    % -------------------------------------------
    % By shaoguangcheng. From Xi'an, China.
    % Email : chengshaoguang1291@126.com
    
    properties
        className = 'cdbn';
        
        nLayer;        % total number of layers in CDBN
        model;         % all layer paramters and structures
        output;        % output of cdbn
        outputFolder;
        isSaveModel = true;
    end
    
    methods
        function self = cdbn(netStructure)
            % ---------------------------
            % create cdbn
            % ---------------------------
           self.nLayer = numel(netStructure);
           self.model = cell(1, self.nLayer);
           
           % initialize all layers in cdbn
           self.model{1} = crbm(netStructure(1));
           if self.model{1}.nFeatureMapVis ~= 1
              error('First layer in cdbn must only have single feature map'); 
           end
           
           if self.nLayer > 1
               for i = 2 : self.nLayer
                  self.model{i} = crbm(netStructure(i)); 
               end
           end
           
            self.outputFolder = sprintf('%s%s%s','..', filesep, 'log');
            if ~exist(self.outputFolder, 'dir')
                mkdir(self.outputFolder);
            end
           
        end
        
        function self = train(self, data)
            % ----------------------
            % train cdbn model
            % ----------------------
             self.model{1} = self.model{1}.train(data);
             self.model{1} = self.model{1}.crbmFeedForward(data);            
            
            for i = 2 : self.nLayer
                self.model{i} = self.model{i}.train(self.model{i-1}.outputPooling);
                self.model{i} = self.model{i}.crbmFeedForward(self.model{i-1}.outputPooling);
            end
            
            if self.isSaveModel
                
            end
        end
        
        function self = cdbnFeedForward(self, data)
           for i = 1 : self.nLayer
               data = self.model{i}.crbmFeedForward(data);
           end
           self.output = data;
        end
        
        function [] = save(self)
            fmt = sprintf('%s%s%s_cdbn_model.mat',self.outputFolder, ...
                filesep, datestr(clock, 'yyyy_mm_dd_HH_MM_SS')); 
            
            cdbnModel = cell(1, self.nLayer);
            for i  = 1 : self.nLayer
               cdbnModel{i}.W = self.model{i}.W;
               cdbnModel{i}.visBias = self.model{i}.visBias;
               cdbnModel{i}.hidBias = self.model{i}.hidBias;
            end
            
            save(fmt, cdbnModel);
        end
        
    end
    
end

