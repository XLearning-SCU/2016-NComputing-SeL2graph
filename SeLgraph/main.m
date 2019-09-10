% =========================================================================
% An example code for the algorithm proposed in

%   [1] Xi Peng, Miaolong Yuan, Zhiding Yu, Wei Yun Yau and Lei Zhang,
%       Semi-supervised Subspace learning with L2-Graph
%       Neurocomputing, Accepted     

% The method [1] is a semi-supervised extension of L2-Graph
% proposed in 

%   [2] Xi Peng, Zhang Yi, and Huajin Tang.
%       Robust Subspace Clustering via Thresholding Ridge Regression.
%       The Twenty-Ninth AAAI Conference on Artificial Intelligence (AAAI), Austin, Texas, USA, January 25–29, 2015.

%   [3] Xi Peng, Zhiding Yu, Zhang Yi, and Huajin Tang.
%       Constructing the L2-Graph for Robust Subspace Learning and Subspace Clustering.
%       IEEE Trans. On Cybernetics, accepted.

% Written by Xi Peng @ I2R A*STAR
% Aug., 2015.

% Description: semi-supervised L2-Graph.
% Each column corresponds to a data point.
% more information can be accessed at www.pengxi.me

% ***NOTICED that***:
% If the codes or data sets are helpful to you, please appropriately cite our works. Thank you very much!
% =========================================================================

close all;
clear all;
clc;
% --------------------------------------------------------------------------
addpath ('../usages/');
addpath ('../data/');
addpath ('../classifers/');
addpath ('../classifers/SVM/');

% ---------- loading data
CurData = 'AR_55_40_700vs700';
load (CurData);  
% ---------- data option parameters configuration
options.nClass             = 100;  % the first nClass subjects are tested
options.nDim               = 2200; %input dimensionality corresponding to the cropped size
% ---------- similarity graph optionsameters configuration
options.lambda             = [0.1];% L2 regularization parameter 
options.adjKnn             = [20];  % parameter to truncate the trival coefs.

% % ---------- loading data
% CurData = 'ExYaleB_54_48_1102vs1102';
% load (CurData);  
% % ---------- data option parameters configuration
% options.nClass             = 39;  % the first nClass subjects are tested
% options.nDim               = 2592; %input dimensionality corresponding to the cropped size
% % ---------- similarity graph optionsameters configuration
% options.lambda             = [0.1];% L2 regularization parameter 
% options.adjKnn             = [14];  % parameter to truncate the trival coefs.


% % ---------- loading data
% CurData = 'FERET_800vs600';
% load (CurData);  
% % ---------- data option parameters configuration
% options.nClass             = 200;  % the first nClass subjects are tested
% options.nDim               = 6400; %input dimensionality corresponding to the cropped size
% % ---------- similarity graph optionsameters configuration
% options.lambda             = [0.3];% L2 regularization parameter 
% options.adjKnn             = [16];  % parameter to truncate the trival coefs.
% 

% % ---------- loading data
% CurData = 'MPIES1_7vs7';
% load (CurData);  
% % ---------- data option parameters configuration
% options.nClass             = max(trainlabels);  % the first nClass subjects are tested
% options.nDim               = 2050; %input dimensionality corresponding to the cropped size
% % ---------- similarity graph optionsameters configuration
% options.lambda             = [0.1];% L2 regularization parameter 
% options.adjKnn             = [6];  % parameter to truncate the trival coefs.


% % ---------- loading data
% CurData = 'USPS_5500_550';
% load (CurData);  
% % ---------- data option parameters configuration
% options.nClass             = 100;  % the first nClass subjects are tested
% options.nDim               = 256; %input dimensionality corresponding to the cropped size
% % ---------- similarity graph optionsameters configuration
% options.lambda             = [0.1];% L2 regularization parameter 
% options.adjKnn             = [10];  % parameter to truncate the trival coefs.




% % ---------- loading data
% CurData = 'ExYaleB_54_48_SCRate50_GaussianRate10_1102vs1102';
% load (CurData);  
% % ---------- data option parameters configuration
% options.nClass             = max(trainlabels);  % the first nClass subjects are tested
% options.nDim               = 2592; %input dimensionality corresponding to the cropped size
% % ---------- similarity graph optionsameters configuration
% options.lambda             = [0.1];% L2 regularization parameter 
% options.adjKnn             = [15];  % parameter to truncate the trival coefs.


% % ---------- loading data
% CurData = 'ExYaleB_54_48_SCRate50_PixCorruptionRate10_1102vs1102';
% load (CurData);  
% % ---------- data option parameters configuration
% options.nClass             = max(trainlabels);  % the first nClass subjects are tested
% options.nDim               = 2592; %input dimensionality corresponding to the cropped size
% % ---------- similarity graph optionsameters configuration
% options.lambda             = [0.1];% L2 regularization parameter 
% options.adjKnn             = [15];  % parameter to truncate the trival coefs.


% % ---------- loading data
% CurData = 'AR_glass_permute_600vs600';
% load (CurData);  
% % ---------- data option parameters configuration
% options.nClass             = 100;  % the first nClass subjects are tested
% options.nDim               = 2200; %input dimensionality corresponding to the cropped size
% % ---------- similarity graph optionsameters configuration
% options.lambda             = [0.1];% L2 regularization parameter 
% options.adjKnn             = [3];  % parameter to truncate the trival coefs.


% % ---------- loading data
% CurData = 'AR_scarve_permute_600vs600';
% load (CurData);  
% % ---------- data option parameters configuration
% options.nClass             = 100;  % the first nClass subjects are tested
% options.nDim               = 2200; %input dimensionality corresponding to the cropped size
% % ---------- similarity graph optionsameters configuration
% options.lambda             = [0.1];% L2 regularization parameter 
% options.adjKnn             = [3];  % parameter to truncate the trival coefs.


% ---------- discriminative parameter 
options.ReguBeta           = [0.1];% Parameter to tune the weight between supervised info and local info
options.nRep               = 10;   % the number of repeating tests 
% -------------- preprocess the data
% ----------preprocess with PCA if size(DAT,1) > options.nDim
% ----------perform test using the first option.nClass subjects
NewTrain_DAT = double(NewTrain_DAT);
NewTest_DAT = double(NewTest_DAT);
trls = double(trainlabels);
ttls = double(testlabels);
clear testlabels trainlabels;

options.gnd = trls;
options.NameStr = ['DisL2Graph_' CurData '_Class' num2str(options.nClass) '_PCAdim' num2str(options.nDim) '_lambda#' num2str(length(options.lambda)) '_adjknn#' num2str(length(options.adjKnn)) '_nRep' num2str(options.nRep)];

% ------- perform coding, embedding, and classification
% the labeled data used for providing label information
semiSplit = logical(ones(length(trls),1));
% semiSplit = semidentifier(trls,7);
for i = 1:options.nRep   
    tic;
    % ------ building the similarity graph using L2-Graph
    options.W = BuildingL2Graph(NewTrain_DAT, options.lambda, options.adjKnn);
    fprintf(['+built the graph using ' num2str(options.adjKnn) ' largest coef.!\n']);
    fprintf(['|lambda=' num2str(options.lambda) ' and adjKnn=' num2str(options.adjKnn) '\n']);
    Graph_tElapsed(i) = toc;
    % ------ embedding
    tic;
    [eigvector, eigvalue] = Embedding(reshape(trls,[],1),NewTrain_DAT,semiSplit,options);    
    fprintf('|embeded the principle components graph, finished!\n\n');
    %  ------ dimension reduction 
    tr_dat = eigvector'*NewTrain_DAT;
    tt_dat = eigvector'*NewTest_DAT;
    DR_tElapsed(i) = toc;
    %  ------ classification
    tic;
    SRC_rec(i)  = SRC(tr_dat, tt_dat, trls, ttls);% note that, SRC will achieve better result by enforcing CKSym = abs(CKSym). 
    SRC_time(i) = Graph_tElapsed(i)+DR_tElapsed(i)+toc;

    tic;
    SVM_rec(i) = SVM(tr_dat, tt_dat, trls, ttls);
    SVM_time(i) = Graph_tElapsed(i)+DR_tElapsed(i)+toc;

    tic;
    KNN_rec(i) = NN(tr_dat, tt_dat, trls, ttls);    
    KNN_time(i) = Graph_tElapsed(i)+DR_tElapsed(i)+toc;
end;


clear ans Predict_label kk trls ttls tr_dat DAT tt_dat CurRepeat;
clear eigvector eigvalue;
clear j coef CKSym i t_1nn_ac;
clear NewTest_DAT NewTrain_DAT semiSplit options.gnd options.W;
AnalyzeResult_Rep