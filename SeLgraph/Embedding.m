function [eigvector, eigvalue] = Embedding(gnd,fea,semiSplit,options)

%       [eigvector, eigvalue] = SDA(gnd,feaLabel,feaUnlabel,options)
% 
%             Input:
%               gnd     - Label vector.  
%               fea     - data matrix. Each row vector of fea is a data point. 
%
%             semiSplit - fea(semiSplit,:) is the labeled data matrix. 
%                       - fea(~semiSplit,:) is the unlabeled data matrix. 
%
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%
%                      W           Similarity graph.
%
%                     ReguBeta     Parameter to tune the weight between
%                                  supervised info and local info
%                                    Default 0.1. 
%                                       
%                     ReguAlpha    Paramter of Tinkhonov regularizer
%                                    Default 0.1. 
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = eigvector'*x
%                           will be the embedding result of x.
%               eigvalue  - The eigvalue of eigen-problem. sorted from
%                           largest to smallest. 


if ~isfield(options,'ReguType')
    options.ReguType = 'Ridge';
end
if ~isfield(options,'ReguAlpha')
    options.ReguAlpha = 0.1;
end


[nSmp,nFea] = size(fea');
nSmpLabel = sum(semiSplit);
nSmpUnlabel = sum(~semiSplit);


if nSmpLabel+nSmpUnlabel ~= nSmp
    error('input error!');
end

if ~isfield(options,'W') 
    options.WOptions.gnd = gnd;
    options.WOptions.semiSplit = semiSplit;
    W = constructW(fea,options.WOptions);
else
    W = options.W;
end

gnd = gnd(semiSplit);

classLabel = unique(gnd);
nClass = length(classLabel);
Dim = nClass;

% % ----------------covariance matrix----------------
W = eye(size(W,1)) - W;
W = W * W';
% % ----------------label matrix----------------

ReguBeta = 0.1;
if isfield(options,'ReguBeta') && (options.ReguBeta > 0) 
    ReguBeta = options.ReguBeta;
end
Omega = zeros(nSmp);
for i=1:nClass
    ind = find(classLabel(i) == gnd);
    n_i = length(ind);    
    Omega(ind,ind) = 1/n_i;    
end
% DPrime = W + ReguBeta*(eye(nSmp) - Omega);
DPrime = ReguBeta*W + (eye(nSmp) - Omega); % it is equivalent to  W + 1/ReguBeta*(eye(nSmp) - Omega);

DPrime = fea * DPrime * fea';


switch lower(options.ReguType)
    case {lower('Ridge')}
        for i=1:size(DPrime,1)
            DPrime(i,i) = DPrime(i,i) + options.ReguAlpha;
        end
    case {lower('RidgeLPP')}
        for i=1:size(DPrime,1)
            DPrime(i,i) = DPrime(i,i) + options.ReguAlpha;
        end
    case {lower('Tensor')}
        DPrime = DPrime + options.ReguAlpha*options.regularizerR;
    case {lower('Custom')}
        DPrime = DPrime + options.ReguAlpha*options.regularizerR;
    otherwise
        error('ReguType does not exist!');
end
DPrime = max(DPrime,DPrime');

% ------------
WPrime = fea * Omega * fea';

dimMatrix = size(WPrime,1);

if Dim > dimMatrix
    Dim = dimMatrix; 
end


if isfield(options,'bEigs')
    if options.bEigs
        bEigs = 1;
    else
        bEigs = 0;
    end
else
    if (dimMatrix > 1000 && Dim < dimMatrix/20)  
        bEigs = 1;
    else
        bEigs = 0;
    end
end


if bEigs
    %disp('use eigs to speed up!');
    option = struct('disp',0);
    [eigvector, eigvalue] = eigs(WPrime,DPrime,Dim);
    eigvalue = diag(eigvalue);
else
    [eigvector, eigvalue] = eig(WPrime,DPrime);
    eigvalue = diag(eigvalue);
    
    [junk, index] = sort(-eigvalue);
    eigvalue = eigvalue(index);
    eigvector = eigvector(:,index);

    if Dim < size(eigvector,2)
        eigvector = eigvector(:, 1:Dim);
        eigvalue = eigvalue(1:Dim);
    end
end

for i = 1:size(eigvector,2)
    eigvector(:,i) = eigvector(:,i)./norm(eigvector(:,i));
end


