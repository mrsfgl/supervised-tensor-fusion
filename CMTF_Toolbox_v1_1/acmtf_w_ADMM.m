function [Zhat, G, output] = acmtf_w_ADMM(Z, R, varargin)
%
%


%% Error checking
cmtf_check(Z);

if (nargin < 2)
    error('Error: invalid input arguments');
end

%% Set parameters
params = inputParser;
params.addParameter('beta_cp', 0, @(x) x >= 0);
params.addParameter('beta_pca',0, @(x) x >= 0);
params.addParameter('alpha',0, @(x) x >= 0);
params.addParameter('eta',0, @(x) x >= 0);
params.addParameter('theta',0, @(x) x >= 0);
params.addParameter('init', 'random', @(x) (isstruct(x) || ismember(x,{'random','nvecs'})));
params.parse(varargin{:});
P = numel(Z.object);

%% Initialization
sz = Z.size;
modes = cell2mat(Z.modes);
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

%% Fit ACMTF using ADMM
Znormsqr = cell(P,1);
for p = 1:P
    if isa(Z.object{p},'tensor') || isa(Z.object{p},'sptensor')
        Znormsqr{p} = norm(Z.object{p})^2;
    else
        Znormsqr{p} = norm(Z.object{p},'fro')^2;
    end
end
Zhat = acmtf_fun_admm(G, Z, Znormsqr, params.Results.beta_cp,...
    params.Results.beta_pca, params.Results.alpha, params.Results.eta,...
    params.Results.theta);

end