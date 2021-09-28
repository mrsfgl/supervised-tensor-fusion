function [Zhat, out, G0]=extract_w_CMTF(I1, I2, R, varargin)
%EXTRACT_W_CMTF 
%

params = inputParser;
params.addParameter('modes', {[1,2,3], [4,3]},@(x) iscell(x));
params.parse(varargin{:});

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

%% Fit CMTF using one of the first-order optimization algorithms 
options = ncg('defaults');
options.Display ='off';
options.MaxFuncEvals = 100000;
options.MaxIters     = 500;
options.StopTol      = 1e-7;
options.RelFuncTol   = 1e-7;
options.TraceFunc    = true;

init = 'random';
% fit CMTF-OPT
[Zhat,G0,out] = cmtf_opt(Z,R,'init',init,'alg_options',options); 

end