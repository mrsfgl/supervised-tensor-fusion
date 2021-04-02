function  data = tester_acmtf_admm_missing(varargin)
% TESTER_ACMTF_MISSING Generates coupled data sets with missing entries based 
% on the optional input parameters and then uses ACMTF_OPT to fit a CMTF
% model. 
%
%   data = tester_acmtf_missing; uses default settings to generate coupled data sets
%   and returns data, which is a structure with fields:
%           Zhat: factor matrices extracted using coupled matrix and tensor
%                 factorization, ktensor Zhat{p} has the weights of
%                 components and the factor matrices for the pth object.
%           W    : missing data indicator for each data set
%           Xorig: Original data sets constructed using the true factor
%                  matrices in data.Atrue
%           Init : Initialization used for the optimization algorithm.
%           out  : output of the optimization showing the stopping
%                  condition, gradient, etc.
%           Atrue: factor matrices used to generate data
%           lambdas, lambdas_rec: weights used to generate data and weights
%                                 extracted using ACMTF, respectively.
%           adjlambda_rec : lambdas_rec multiplied by the Frobenius norm of 
%                           each data set.
%           norms: Frobenius norm of each data set used to scale each data
%                  set (once the missing entries set to 0).
%
%   data = tester_acmtf_missing('R',...) gives the number of components that 
%   will be extracted from coupled data sets.
%
%   data = tester_acmtf_missing('modes',...) gives as input how the modes are
%   coupled among each data set, e.g., {[1 2 3], [1 4], [2 5]}- generates
%   three data sets, where the third-order tensor shares the first mode with
%   the first matrix and the second mode with the second matrix.
%
%   data = tester_acmtf_missing('size',...) gives as input the size of the data
%   sets, e.g., if we were to generate a third-order tensor of size 50 by 40
%   by 30 and a matrix of size 50 by 10 coupled in the first mode, then size
%   will be [50 40 30 10].
%
%   data = tester_acmtf_missing('lambdas',...) gives as input the weights of each
%   component in each data set, e.g., {[1 1 0], [1 1 1]}, the first two
%   components are shared by both data sets while the last component is only
%   available in the second data. 
% 
%   data = tester_acmtf_missing('init',...) gives as input the type of
%   initialization to be used for the optimization algorithms,
%   e.g.,'random', 'nvecs', or a structure with fields fac and norms.
%
%   data = tester_acmtf_missing('beta_cp',...,'beta_pca') gives as input the 
%   sparsity penalty parameters.
%
%   data = tester_acmtf_missing('flag_sparse',...) indicates whether each data set will 
%   be stored in the dense or sparse tensor format.
%
%   data = tester_acmtf_missing('M',...) gives as input the percentage of missing entries
%   for each data set, e.g., [0.5 0.5].
%
% See also ACMTF_OPT, CREATE_COUPLED, TT_CREATE_MISSING_DATA_PATTERN
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
params.addParamValue('R', 3, @(x) x > 0);
params.addParamValue('size', [50 30 40 20], @isnumeric);
params.addParamValue('beta_cp', 0, @(x) x >= 0);
params.addParamValue('beta_pca', 0, @(x) x >= 0);
params.addParamValue('alpha', 0, @(x) x >= 0);
params.addParamValue('eta', 0, @(x) x >= 0);
params.addParamValue('theta', 0, @(x) x >= 0);
params.addParamValue('modes', {[1 2 3], [1 4]}, @iscell);
params.addParamValue('lambdas', {[1 1 1], [1 1 1]}, @iscell);
params.addParamValue('M',[0.5 0.5], @isnumeric);
params.addParamValue('flag_sparse',[0 0], @isnumeric);
params.addParamValue('init', 'random', @(x) (isstruct(x) || ismember(x,{'random','nvecs'})));
params.parse(varargin{:});

%% Parameters
lambdas     = params.Results.lambdas;
modes       = params.Results.modes;
sz          = params.Results.size;
R           = params.Results.R;
init        = params.Results.init;
beta_cp     = params.Results.beta_cp;
beta_pca    = params.Results.beta_pca;
alpha       = params.Results.alpha;
eta         = params.Results.eta;
theta       = params.Results.theta;
M           = params.Results.M;
flag_sparse = params.Results.flag_sparse;

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
[X, Atrue] = create_coupled('size', sz, 'modes', modes, 'lambdas', lambdas);

P = length(X);
for p=1:P        
    W{p}  = tt_create_missing_data_pattern(sz(modes{p}), M(p), flag_sparse(p));
    if flag_sparse(p)
        Z.object{p} = W{p}.*sptensor(X{p});
    else
        Z.object{p} = W{p}.*X{p};
    end
    norms(p)    = norm(Z.object{p});
    Z.object{p} = Z.object{p}/norms(p); 
    Z.miss{p}   = W{p};
end
Z.modes = modes;
Z.size  = sz;


% fit ACMTF-OPT
Zhat = acmtf_w_ADMM(Z,R,'init', init, 'beta_cp', beta_cp, 'beta_pca', beta_pca, 'alpha', alpha, 'eta', eta, 'theta', theta);        
data.Zhat  = Zhat;
data.W     = W;
data.Xorig = X;
data.Atrue = Atrue;
l_rec = zeros(P, R);
for p = 1:P
    temp        = normalize(Zhat{p});
    l_rec(p,:)  = temp.lambda;    
end
tt = [];
for i  = 1:length(lambdas)
    tt = [tt; lambdas{i}];
end
data.lambdas       = tt;
data.lambda_rec    = l_rec;
data.adjlambda_rec =  khatrirao(norms,l_rec');
data.norms         = norms;
