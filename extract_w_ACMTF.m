function [Zhat, out]=extract_w_ACMTF(I1, I2, R, beta_cp, beta_pca)
%
%

W{1}  = tt_create_missing_data_pattern(size(I1), 0);
Z.object{1} = tensor(I1).*W{1};
Z.miss{1} = W{1};
W{2}  = tt_create_missing_data_pattern(size(I2), 0);
Z.object{2} = tensor(I2).*W{2};
Z.miss{2} = W{2};
Z.modes = {[1,2,3],[4,3]};
Z.size  = [size(I1),size(I2,1)];

%% Fit ACMTF using one of the first-order optimization algorithms 
options = ncg('defaults');
options.Display ='final';
options.MaxFuncEvals = 100000;
options.MaxIters     = 1000;
options.StopTol      = 1e-6;
options.RelFuncTol   = 1e-6;

init = 'random';
% fit ACMTF-OPT
[Zhat,~,out] = acmtf_opt(Z,R,'init',init,'alg_options',options, 'beta_cp', beta_cp, 'beta_pca', beta_pca); 

end