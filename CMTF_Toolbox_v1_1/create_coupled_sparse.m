function  [X, A] = create_coupled_sparse(varargin)
% CREATE_COUPLED_SPARSE generates coupled sparse higher-order tensors and matrices 
% and returns the generated data as a cell array, X, the factors used to generate 
% these data sets as a cell array, A. 
% 
%   [X, A] = create_coupled_sparse('size',...) gives as input the size of the data
%   sets, e.g., if we were to generate a third-order tensor of size 50 by 40
%   by 30 and a matrix of size 50 by 10 coupled in the first mode, then size
%   will be [50 40 30 10].
%
%   [X, A] = create_coupled_sparse('modes',...) gives as input how the modes are
%   coupled among each data set, e.g., {[1 2 3], [1 4], [2 5]}- generates
%   three data sets, where the third-order tensor shares the first mode with
%   the first matrix and the second mode with the second matrix.
%
%   [X, A] = create_coupled_sparse('lambdas', ...) gives as input the norms of each
%   component in each data set, e.g., {[1 1 0], [1 1 1]}, the first two
%   components are shared by both data sets while the last component is only
%   available in the second data.
%
%   [X, A] = create_coupled_sparse('noise',....) gives as input the noise level 
%   (random entries following the standard normal) to be added to each data set.
%
%   [X, A] = create_coupled_sparse('flag_sparse',....) gives as input the
%   indicator showing whether each data set will be in the dense or sparse
%   format, e.g., [0 1] indicates that the first data set is in dense
%   format while the second one is in the sptensor format.
%
% See also TESTER_ACMTF, TESTER_CMTF
%
% This is the MATLAB CMTF Toolbox, 2013.
% References: 
%    - (CMTF) E. Acar, T. G. Kolda, and D. M. Dunlavy, All-at-once Optimization for Coupled
%      Matrix and Tensor Factorizations, KDD Workshop on Mining and Learning
%      with Graphs, 2011 (arXiv:1105.3422v1)
%    - (ACMTF)E. Acar, A. J. Lawaetz, M. A. Rasmussen,and R. Bro, Structure-Revealing Data 
%      Fusion Model with Applications in Metabolomics, IEEE EMBC, pages 6023-6026, 2013.
%    - (ACMTF)E. Acar,  E. E. Papalexakis, G. Gurdeniz, M. Rasmussen, A. J. Lawaetz, M. Nilsson, and R. Bro, 
%      Structure-Revealing Data Fusion, BMC Bioinformatics, 15: 239, 2014.        
%

%% Parse inputs
params = inputParser;
params.addParamValue('size', [50 30 40 20], @isnumeric);
params.addParamValue('modes', {[1 2 3], [1 4]}, @iscell);
params.addParamValue('noise', 0.1, @(x) x >= 0);
params.addParamValue('lambdas', {[1 1 1], [1 1 1]}, @iscell);
params.addParamValue('flag_sparse',[0 0], @isnumeric);
params.parse(varargin{:});
sz          = params.Results.size;    %size of data sets
lambdas     = params.Results.lambdas; % norms of components in each data set
modes       = params.Results.modes;   % how the data sets are coupled
nlevel      = params.Results.noise;
flag_sparse = params.Results.flag_sparse;
max_modeid  = max(cellfun(@(x) max(x), modes));
if max_modeid ~= length(sz)
    error('Mismatch between size and modes inputs')
end
P  = length(modes);

%% Generate factor matrices
nb_modes  = length(sz);
Rtotal    = length(lambdas{1});
A         = cell(nb_modes,1);
% generate factor matrices
for n = 1:nb_modes
    A{n} = randn(sz(n),Rtotal);
    for r=1:Rtotal
        A{n}((find(abs(A{n}(:,r))<0.85)),r)=0;        
        while nnz(A{n}(:,r))==0
            A{n}(:,r) = randn(sz(n),1);
            A{n}((find(abs(A{n}(:,r))<0.85)),r)=0;        
        end
        A{n}(:,r)=A{n}(:,r)/norm(A{n}(:,r));
    end
end

%% Generate data blocks 
X  = cell(P,1);
for p = 1:P        
    if flag_sparse(p)
        X{p} = sptensor(full(ktensor(lambdas{p}', A(modes{p}))));                
        Rdm  = sptensor(X{p}.subs,randn(size(X{p}.subs,1),1),sz(modes{p}));    
        X{p} = X{p} + nlevel* norm(X{p}) * Rdm / norm(Rdm);
    else
        X{p} = full(ktensor(lambdas{p}',A(modes{p})));
        Rdm  = tensor(randn(size(X{p})));
        X{p} = X{p} + nlevel*norm(X{p})*Rdm/norm(Rdm);
    end
end

