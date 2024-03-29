function  data = tester_cmtf_missing(varargin)
% TESTER_CMTF_MISSING Generates coupled data sets with missing entries
% based on the optional input parameters and then uses CMTF_OPT to fit a 
% coupled matrix-tensor factorization model. 
%
%   data = tester_cmtf_missing; uses default settings to generate coupled data sets
%   and returns data, which is a structure with fields:
%           Fac    : a cell array of factor matrices extracted using CMTF
%           W      : missing data indicator for each data set
%           Xorig  : Original data sets constructed using the true factor
%                    matrices in data.Factrue
%           Init   : Initialization used for the optimization algorithm 
%           out    : output of the optimization showing the stopping
%                    condition, gradient, function value, etc.
%           Factrue: a cell array of factor matrices used to generate data
%           norms  : Frobenius norm of each data set used to scale each data
%                    set (once the missing entries set to 0).
%
%   data = tester_cmtf_missing('R',...) gives the number of components that will be
%   extracted from coupled data sets.
%
%   data = tester_cmtf_missing('modes',...) gives as input how the modes are
%   coupled among data sets, e.g., {[1 2 3], [1 4], [2 5]}- generates
%   three data sets, where the third-order tensor shares the first mode with
%   the first matrix and the second mode with the second matrix.
%
%   data = tester_cmtf_missing('size',...) gives as input the size of the data
%   sets, e.g., if we were to generate a third-order tensor of size 50 by 40
%   by 30 and a matrix of size 50 by 10 coupled in the first mode, then size
%   will be [50 40 30 10].
%
%   data = tester_cmtf_missing('lambdas',...) gives as input the weights of each
%   component in each data set, e.g., {[1 1 1], [1 1 1]}. The length of each
%   cell should be the same.
% 
%   data = tester_cmtf_missing('init',...) gives as input the type of initialization
%   to be used for the optimization algorithms, e.g.,'random', 'nvecs', or a 
%   cell array of matrices.
%
%   data = tester_cmtf_missing('flag_sparse',...) indicates whether each data set will 
%   be stored in the dense or sparse tensor format.
%
%   data = tester_cmtf_missing('M',...) gives as input the percentage of missing entries
%   for each data set, e.g., [0.5 0.5].
%
% See also CMTF_OPT, CREATE_COUPLED, TT_CREATE_MISSING_DATA_PATTERN
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
params.addParameter('R', 3, @(x) x > 0);
params.addParameter('size', [50 30 40 20], @isnumeric);
params.addParameter('modes', {[1 2 3], [1 4]}, @iscell);
params.addParameter('lambdas', {[1 1 1], [1 1 1]}, @iscell);
params.addParameter('flag_sparse',[0 0], @isnumeric);
params.addParameter('flag_soft',0, @isnumeric);
params.addParameter('dist_coupled', 0.1, @(x) x >= 0);
params.addParameter('rnd_seed', randi(10^3));
params.addParameter('M',[0.5 0.5], @isnumeric);
params.addParameter('noise',.1, @isnumeric);
params.addParameter('init', 'random', @(x) (iscell(x) || ismember(x,{'random','nvecs'})));
params.parse(varargin{:});

%% Parameters
lambdas     = params.Results.lambdas;
modes       = params.Results.modes;
sz          = params.Results.size;
R           = params.Results.R;
init        = params.Results.init;
flag_sparse = params.Results.flag_sparse;
M           = params.Results.M;
noise       = params.Results.noise;
flag_soft   = params.Results.flag_soft;
dist_coupled = params.Results.dist_coupled;
rnd_seed    = params.Results.rnd_seed;

if length(lambdas{1})~=R
    for i = 1:length(lambdas)
        lambdas{i} = ones(1,R);
    end
end

%% Check parameters
if length(lambdas)~=length(modes)
    error('There should be weights for each data set');
end
P = length(modes);
for p=1:P
    l(p) = length(lambdas{p});
end
if length(unique(l))>1
    error('There should be the same number of weights for each data set');
end
if length(flag_sparse)<p
    error('flag_sparse should be specified for each data set');
end
if length(M)<p
    error('Percentage of missing data should be specified for each data set');
end


%% Form coupled data
if flag_soft
    [X, Atrue] = create_soft_coupled('size',sz,'modes',modes,'lambdas',...
        lambdas,'dist_coupled', dist_coupled,'rnd_seed', rnd_seed, 'noise',...
        noise);
else
    [X, Atrue] = create_coupled('size',sz,'modes',modes,'lambdas',lambdas,...
        'rnd_seed', rnd_seed, 'noise',noise);
end

% Set some entries to missing based on the percentage of missing entries, M.
P = length(X);
for p=1:P
    W{p}  = tt_create_missing_data_pattern(sz(modes{p}), M(p), flag_sparse(p));
    if flag_sparse(p)
        Z.object{p} = W{p}.*sptensor(X{p});
    else
        Z.object{p} = W{p}.*X{p};
    end
    norms(p)    = norm(Z.object{p});
    Z.object{p} = Z.object{p}/norm(Z.object{p});    
    Z.miss{p}   = W{p};
    Xorig{p}    = full(ktensor(lambdas{p}',Atrue(modes{p})));
end
Z.modes = modes;
Z.size  = sz;

%% Fit CMTF using one of the first-order optimization algorithms 
options = ncg('defaults');
options.Display ='off';
options.MaxFuncEvals = 100000;
options.MaxIters     = 500;
options.StopTol      = 1e-6;
options.RelFuncTol   = 1e-6;

% fit CMTF-OPT
[Fac, G, out] = cmtf_opt(Z,R,'init',init,'alg_options',options); 
for p=1:P
    data.Fac(modes{p}) = Fac{p}.U;
end
data.W        = W;
data.X        = X;
data.Xorig    = Xorig;
data.Init     = G; 
data.out      = out.OptOut;
data.Factrue  = Atrue;
data.norms    = norms;
