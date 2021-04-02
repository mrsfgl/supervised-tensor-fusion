function [Zhat, G, output] = acmtf_opt(Z,R,varargin)
% ACMTF_OPT Fits a coupled matrix and tensor factorization (CMTF) model using 
% first-order optimization (with the option of adding sparsity penalties on the
% weights of components).
%
%   Zhat = ACMTF_OPT(Z,R) fits an R-component CMTF model to the coupled 
%   data stored in Z and returns the factor matrices for each in Zhat as 
%   a ktensor. Z is a structure with object, modes, size, miss fields storing 
%   the coupled data sets (See cmtf_check)
%
%   Zhat = ACMTF_OPT(Z,R,'param',value,...) specifies additional
%   parameters for the method. Specifically...
%
%   'alg' - Specifies optimization algorithm {'ncg'}
%      'ncg'   Nonlinear Conjugate Gradient Method
%      'lbfgs' Limited-Memory BFGS Method
%      'tn'    Truncated Newton
%
%   'init' - Initialization for component matrices {'random'}. This
%   can be a structure with fac (cell array) and norms (cell array) fields 
%   storing the factors matrices and the weights to be used for initialization 
%   or one of the following strings:
%      'random' Randomly generated via randn function
%      'nvecs'  Selected as leading left singular vectors of unfolded data 
%               in Z.object{n}
%
%   'alg_options' - Parameter settings for selected optimization
%   algorithm. For example, type OPTIONS = NCG('defaults') to get
%   the NCG algorithm options which can then be modified and passed
%   through this function to NCG.
%
%   'beta_cp'       - Sparsity parameter on the weights of rank-one tensors
%   in the CP part of a CMTF model.
%
%   'beta_pca'      - Sparsity parameter on the weights of rank-one matrices
%   in the PCA part of a CMTF model.
%
%   [Zhat, G0]      = ACMTF_OPT(...) also returns the initial guess.
%
%   [Zhat, G0, OUT] = ACMTF_OPT(...) also returns a structure with the
%   optimization exit flag, the final relative fit, and the full
%   output from the optimization method.
%
% See also ACMTF_FUN, ACMTF_FG, ACMTF_VEC_TO_STRUCT, ACMTF_STRUCT_TO_VEC,
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
params.addParamValue('alg', 'ncg', @(x) ismember(x,{'ncg','tn','lbfgs'}));
params.addParamValue('beta_cp', 0, @(x) x >= 0);
params.addParamValue('beta_pca',0, @(x) x >= 0);
params.addParamValue('init', 'random', @(x) (isstruct(x) || ismember(x,{'random','nvecs'})));
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
N = length(sz);

if isstruct(params.Results.init)
    G.fac   = params.Results.init.fac;
    G.norms = params.Results.init.norms;
elseif strcmpi(params.Results.init,'random')
    G.fac = cell(N,1);
    for n=1:N
        G.fac{n} = randn(sz(n),R);
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
        G.fac{n} = cmtf_nvecs(Z,n,R);
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
out = feval(fhandle, @(x)acmtf_fun(x,Z,R,Znormsqr, params.Results.beta_cp, params.Results.beta_pca),  acmtf_struct_to_vec(G), options);

%% Compute factors 
Temp = acmtf_vec_to_struct(out.X, Z, R);
Zhat = cell(P,1);
for p=1:P
    Zhat{p} = ktensor(Temp.norms{p},Temp.fac(Z.modes{p}));
end
output = out;

