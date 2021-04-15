function  [X, A] = create_coupled(varargin)
% CREATE_COUPLED generates coupled higher-order tensors and matrices -
% and returns the generated data as a cell array, X, as well as the factors 
% used to generate these data sets as a cell array, A.
% 
%   [X, A] = create_coupled('size',...) gives as input the size of the data
%   sets, e.g., if we were to generate a third-order tensor of size 50 by 40
%   by 30 and a matrix of size 50 by 10 coupled in the first mode, then size
%   will be [50 40 30 10].
%
%   [X, A] = create_coupled('modes',...) gives as input how the modes are
%   coupled among each data set, e.g., {[1 2 3], [1 4], [2 5]}- generates
%   three data sets, where the third-order tensor shares the first mode with
%   the first matrix and the second mode with the second matrix.
%
%   [X, A] = create_coupled('lambdas', ...) gives as input the norms of each
%   component in each data set, e.g., {[1 1 0], [1 1 1]}, the first two
%   components are shared by both data sets while the last component is only
%   available in the second data.
%
%   [X, A] = create_coupled('noise',....) gives as input the noise level 
%   (random entries following the standard normal) to be added to each data set.
%
% See also TESTER_ACMTF, TESTER_CMTF, TESTER_ACMTF_MISSING, TESTER_CMTF_MISSING
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
params.addParameter('size', [50 30 40 20 10], @isnumeric);
params.addParameter('modes', {[1 2 3], [1 4], [1 5]}, @iscell);
params.addParameter('noise', 0.1, @(x) x > 0);
params.addParameter('lambdas', {[1 1 1], [1 1 1], [1 1 0]}, @iscell);
params.addParameter('rnd_seed', randi(10^3));
params.parse(varargin{:});
sz         = params.Results.size;    %size of data sets
lambdas    = params.Results.lambdas; % norms of components in each data set
modes      = params.Results.modes;   % how the data sets are coupled
nlevel     = params.Results.noise;
rnd_seed   = params.Results.rnd_seed;


max_modeid = max(cellfun(@(x) max(x), modes));
if max_modeid ~= length(sz)
    error('Mismatch between size and modes inputs')
end

%% Random seed
rng(rnd_seed);

%% Generate factor matrices
nb_modes  = length(sz);
Rtotal    = length(lambdas{1});
A         = cell(nb_modes,1);
% generate factor matrices
for n = 1:nb_modes
    A{n} = randn(sz(n),Rtotal);
    for r=1:Rtotal
        A{n}(:,r)=A{n}(:,r)/norm(A{n}(:,r));
    end
end

%% Generate data blocks
P  = length(modes);
X  = cell(P,1);
for p = 1:P
    X{p} = full(ktensor(lambdas{p}',A(modes{p})));
    N    = tensor(randn(size(X{p})));
    X{p} = X{p} + nlevel*norm(X{p})*N/norm(N);
end
