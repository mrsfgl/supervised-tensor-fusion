function  data = tester_acmtf_wsc_missing(varargin)
% TESTER_ACMTF_WSC_MISSING Generates coupled data sets with missing entries 
% based on the optional input parameters and then uses ACMTF_WSC to fit a CMTF
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

%% Parse inputs
params = inputParser;
params.addParameter('R', 3, @(x) x > 0);
params.addParameter('size', [50 30 40 20], @isnumeric);
params.addParameter('beta_cp', .001, @(x) x >= 0);
params.addParameter('beta_pca', .001, @(x) x >= 0);
params.addParameter('modes', {[1 2 3], [1, 4]}, @iscell);
params.addParameter('lambdas', {[1 1 1], [1 1 1]}, @iscell);
params.addParameter('M',[0.5 0.5], @isnumeric);
params.addParameter('noise',.1, @(x) x>=0 && x<=1);
params.addParameter('flag_sparse',[0 0], @isnumeric);
params.addParameter('flag_soft', 1, @isnumeric);
params.addParameter('dist_coupled', 0.1, @(x) x >= 0);
params.addParameter('rnd_seed', randi(10^3));
params.addParameter('init', 'random', @(x) (isstruct(x) || ismember(x,{'random','nvecs'})));
params.parse(varargin{:});

%% Parameters
lambdas     = params.Results.lambdas;
modes       = params.Results.modes;
sz          = params.Results.size;
R           = params.Results.R;
init        = params.Results.init;
beta_cp     = params.Results.beta_cp;
beta_pca    = params.Results.beta_pca;
M           = params.Results.M;
noise = params.Results.noise;
flag_sparse = params.Results.flag_sparse;
flag_soft   = params.Results.flag_soft;
rnd_seed    = params.Results.rnd_seed;
dist_coupled = params.Results.dist_coupled;

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


%% Form data
if flag_soft
    [X, Atrue] = create_soft_coupled('size',sz,'modes',modes,'lambdas',...
        lambdas,'dist_coupled', dist_coupled,'rnd_seed', rnd_seed, 'noise',...
        noise);
else
    [X, Atrue] = create_coupled('size',sz,'modes',modes,'lambdas',lambdas,...
        'rnd_seed', rnd_seed, 'noise',noise);
end

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

%% Fit ACMTF using one of the first-order optimization algorithms 
options = ncg('defaults');
options.Display ='off';
options.MaxFuncEvals = 100000;
options.MaxIters     = 500;
options.StopTol      = 1e-6;
options.RelFuncTol   = 1e-6;

% fit ACMTF-OPT
[Zhat,Init,out] = acmtf_opt_wsc(Z,R,'init',init,'alg_options',...
    options,'beta_cp',beta_cp,'beta_pca',beta_pca);        
data.Zhat  = Zhat;
data.W     = W;
data.Xorig = X;
data.Init  = Init;
data.out   = out;
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
