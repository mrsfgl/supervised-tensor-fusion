function [Zhat, G, output] = acmtf_opt_wsc(Z,R,varargin)
% ACMTF_OPT_WSC Fits a soft coupled matrix and tensor factorization (CMTF) 
% model using similar to ACMTF_OPT.
%
%   Zhat = ACMTF_OPT_WSC(Z,R) fits an R-component CMTF model to the coupled 
%   data stored in Z and returns the factor matrices for each in Zhat as 
%   a ktensor. Z is a structure with object, modes, size, miss fields storing 
%   the coupled data sets (See cmtf_check)
%
%   Zhat = ACMTF_OPT(Z,R,'param',value,...) specifies additional
%   parameters for the method. See ACMTF_OPT.
%
% See also ACMTF_OPT, ACMTF_FUN_SC, ACMTF_FG_WSC, ACMTF_VEC_TO_STRUCT_SC, 
% SCP_FG, SCP_WFG, SPCA_FG, SPCA_WFG, CMTF_CHECK, CMTF_NVECS
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

%% Error checking
cmtf_check(Z);

if (nargin < 2)
    error('Error: invalid input arguments');
end

%% Set parameters
params = inputParser;
params.addParameter('alg', 'ncg', @(x) ismember(x,{'ncg','tn','lbfgs'}));
params.addParameter('beta_cp', 0, @(x) x >= 0);
params.addParameter('beta_pca', 0, @(x) x >= 0);
params.addParameter('alpha', 1, @(x) x >= 0);
params.addParameter('init', 'random', @(x) (isstruct(x) || ismember(x,{'random','nvecs'})));
params.addOptional('alg_options', '', @isstruct);
params.parse(varargin{:});
P = numel(Z.object);

%% Set up optimization algorithm
switch (params.Results.alg)
    case 'ncg'
        fhandle = @ncg;
    case 'tn'
        fhandle = @tn;
    case 'lbfgs'
        fhandle = @lbfgs;
end

%% Set up optimization algorithm options
if isempty(params.Results.alg_options)
    options = feval(fhandle, 'defaults');
else
    options = params.Results.alg_options;
end

%% Initialization
sz = Z.size;
modes = cell2mat(Z.modes);
n_modes = cellfun(@length, Z.modes);
n_modes = cumsum(n_modes);
N = length(modes);

if isstruct(params.Results.init)
    G.fac   = params.Results.init.fac;
    G.norms = params.Results.init.norms;
elseif strcmpi(params.Results.init,'random')
    G.fac = cell(N,1);
    for n=1:N
        G.fac{n} = randn(sz(modes(n)),R);
        for j=1:R
            G.fac{n}(:,j) = G.fac{n}(:,j) / norm(G.fac{n}(:,j));
        end
    end
    for p=1:P
        G.norms{p} = ones(R,1);
    end
elseif strcmpi(params.Results.init,'nvecs')
    G.fac = cell(N,1);
    for n=1:N
        G.fac{n} = cmtf_nvecs(Z,modes(n),R);
    end
    for p=1:P
        G.norms{p} = ones(R,1);
    end
else
    error('Initialization type not supported')
end

%% Fit ACMTF using Optimization
Znormsqr = cell(P,1);
for p = 1:P
    if isa(Z.object{p},'tensor') || isa(Z.object{p},'sptensor')
        Znormsqr{p} = norm(Z.object{p})^2;
    else
        Znormsqr{p} = norm(Z.object{p},'fro')^2;
    end
end
out = feval(fhandle, @(x)acmtf_fun_sc(x,Z,R,Znormsqr, params.Results.beta_cp, params.Results.beta_pca, params.Results.alpha),  acmtf_struct_to_vec(G), options);

%% Compute factors 
Temp = acmtf_vec_to_struct_sc(out.X, Z, R);
Zhat = cell(P,1);
Zhat{1} = ktensor(Temp.norms{1},Temp.fac(1:n_modes(1)));
for p=2:P
    Zhat{p} = ktensor(Temp.norms{p},Temp.fac((n_modes(p-1)+1):n_modes(p)));
end
output = out;

