function [Zhat, out]=extract_wsc_ACMTF(I1, I2, R, varargin)
%EXTRACT_WSC_ACMTF
%

%% Parse inputs
params = inputParser;
params.addParameter('beta_cp', 1e-2, @(x) x >= 0);
params.addParameter('beta_pca', 1e-2, @(x) x >= 0);
params.addParameter('alpha', .01, @(x) x >= 0);
params.addParameter('modes', {[1,2,3], [4,3]},@(x) iscell(x));
params.parse(varargin{:});

beta_cp = params.Results.beta_cp;
beta_pca = params.Results.beta_pca;
alpha = params.Results.alpha;
modes = params.Results.modes;

if isstruct(I1) && isempty(I2)
    Z = I1;
else
    W{1}  = tt_create_missing_data_pattern(size(I1), 0);
    Z.object{1} = tensor(I1).*W{1};
    Z.miss{1} = W{1};
    W{2}  = tt_create_missing_data_pattern(size(I2), 0);
    Z.object{2} = tensor(I2).*W{2};
    Z.miss{2} = W{2};
    Z.modes = modes;
    Z.size = zeros(1,length(unique(cell2mat(modes))));
    modes = cell2mat(modes);
    sizes = cell2mat(cellfun(@size, Z.object, 'UniformOutput', false));
    Z.size(modes) = sizes;
end

%% Fit ACMTF using one of the first-order optimization algorithms 
options = ncg('defaults');
options.Display ='off';
options.MaxFuncEvals = 100000;
options.MaxIters     = 500;
options.StopTol      = 1e-7;
options.RelFuncTol   = 1e-7;
options.TraceFunc    = true;

init = 'random';
% fit ACMTF-OPT
[Zhat,~,out] = acmtf_opt_wsc(Z,R,'init',init,'alg_options',options,...
    'beta_cp', beta_cp, 'beta_pca', beta_pca, 'alpha', alpha); 

end