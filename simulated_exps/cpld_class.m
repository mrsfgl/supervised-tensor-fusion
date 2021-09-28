function results = cpld_class(X, A, K)

n_samples = length(X{1});
n_test = round(n_samples./K);
n_train = n_samples-n_test;
R = size(A{1}{1}{1},2);
modes = X{1}{1}.modes;
mds = cell2mat(X{1}{1}.modes);
P = length(modes);
beta_cp = .001; beta_pca = .001;
alpha = 0.1;

options = ncg('defaults');
options.Display ='off';
options.MaxFuncEvals = 100000;
options.MaxIters     = 500;
options.StopTol      = 1e-7;
options.RelFuncTol   = 1e-7;
options.TraceFunc    = true;

err_orig = zeros(n_samples*2, P); corr_orig = zeros(n_samples*2, length(mds)-1);
err_cp = zeros(n_samples*2, 1); err_pca = zeros(n_samples*2, 1);
err_cmtf = zeros(n_samples*2, P); err_acmtf = zeros(n_samples*2, P);
err_acmtf_sc = zeros(n_samples*2, P);
corr_cp = zeros(n_samples*2, length(modes{1})); corr_pca = zeros(n_samples*2, length(modes{2}));
corr_cmtf = zeros(n_samples*2, length(mds)-1); 
corr_acmtf = zeros(n_samples*2, length(mds)-1);
corr_acmtf_sc = zeros(n_samples*2, length(mds));
fms_cp = zeros(n_samples*2, length(modes{1})); fms_pca = zeros(n_samples*2,length(modes{2}));
fms_cmtf = zeros(n_samples*2, length(mds)-1); fms_acmtf = zeros(n_samples*2, length(mds)-1);
fms_acmtf_sc = zeros(n_samples*2, length(mds)-1); fms_orig = zeros(n_samples*2, length(mds)-1);

Fcp = cell(1, 2*n_samples);
Fpca = Fcp; Fcmtf = Fpca; Facmtf = Fcmtf; Facmtf_sc = Facmtf;
for i=1:n_samples
    %% Extract Factors
%     Fcp{i} = cp_wopt(X{1}{i}.object{1}, X{1}{i}.miss{1}, R);
%     Fpca{i} = cp_wopt(X{1}{i}.object{2}, X{1}{i}.miss{2}, R);
    Fcp{i} = cp_opt(X{1}{i}.object{1}, R, 'alg_options', options);
    Fpca{i} = cp_opt(X{1}{i}.object{2}, R, 'alg_options', options);
    Fcmtf{i} = extract_w_CMTF(X{1}{i}, [], R, 'modes', modes);
    Facmtf{i} = extract_w_ACMTF(X{1}{i}, [], R, 'modes', modes,'beta_cp',...
        beta_cp, 'beta_pca', beta_pca);
%     Facmtf_sc{i} = extract_wsc_ACMTF(X{1}{i},[], R, 'modes', modes,'beta_cp',...
%         beta_cp, 'beta_pca', beta_pca, 'alpha', alpha);
%     Fcp{i+n_samples} = cp_wopt(X{2}{i}.object{1}, X{2}{i}.miss{1}, R);
%     Fpca{i+n_samples} = cp_wopt(X{2}{i}.object{2}, X{2}{i}.miss{2}, R);
    Fcp{i+n_samples} = cp_opt(X{2}{i}.object{1}, R, 'alg_options', options);
    Fpca{i+n_samples} = cp_opt(X{2}{i}.object{2}, R, 'alg_options', options);
    Fcmtf{i+n_samples} = extract_w_CMTF(X{2}{i}, [], R, 'modes', modes);
    Facmtf{i+n_samples} = extract_w_ACMTF(X{2}{i}, [], R, 'modes', modes,...
        'beta_cp', beta_cp, 'beta_pca', beta_pca);
%     Facmtf_sc{i+n_samples} = extract_wsc_ACMTF(X{2}{i}, [], R, 'modes',...
%         modes,'beta_cp', beta_cp, 'beta_pca', beta_pca, 'alpha', alpha);
    
    %% Compute the decomposition quality.
    [err_orig(i,:), corr_orig(i,:), fms_orig(i,:)] = ...
        comp_err_facs(A{1}{i}, A{2}{i}, X{1}{i}.orig, modes);
    err_orig(i+n_samples,:) = err_orig(i,:);
    corr_orig(i+n_samples,:) = corr_orig(i,:);
    [err_cp(i,:), corr_cp(i,:), fms_cp(i,:)] = comp_err_facs(A{1}{i}(modes{1}), Fcp{i}.U,...
        X{1}{i}.orig(1), {1:length(modes{1})});
    [err_cp(i+n_samples,:), corr_cp(i+n_samples,:), fms_cp(i+n_samples,:)] = ...
        comp_err_facs(A{2}{i}(modes{1}), Fcp{i+n_samples}.U, X{2}{i}.orig(1),...
        {1:length(modes{1})});
    [err_pca(i,:), corr_pca(i,:), fms_pca(i,:)] = comp_err_facs(A{1}{i}(modes{2}), Fpca{i}.U,...
        X{1}{i}.orig(2), {1:length(modes{2})});
    [err_pca(i+n_samples,:), corr_pca(i+n_samples,:), fms_pca(i+n_samples,:)] = ...
        comp_err_facs(A{2}{i}(modes{2}), Fpca{i+n_samples}.U, X{2}{i}.orig(2),...
        {1:length(modes{2})});
    [err_cmtf(i,:), corr_cmtf(i,:), fms_cmtf(i,:)] = comp_err_facs(A{1}{i},...
        [Fcmtf{i}{1}.U; Fcmtf{i}{2}.U(1)],X{1}{i}.orig, modes, Fcmtf{i});
    [err_cmtf(i+n_samples,:), corr_cmtf(i+n_samples,:), fms_cmtf(i+n_samples,:)] = ...
        comp_err_facs(A{2}{i}, [Fcmtf{i+n_samples}{1}.U;...
        Fcmtf{i+n_samples}{2}.U(1)], X{2}{i}.orig, modes, Fcmtf{i+n_samples});
    [err_acmtf(i,:), corr_acmtf(i,:), fms_acmtf(i,:)] = comp_err_facs(A{1}{i},...
        [Facmtf{i}{1}.U; Facmtf{i}{2}.U(1)], X{1}{i}.orig, modes, Facmtf{i});
    [err_acmtf(i+n_samples,:), corr_acmtf(i+n_samples,:), fms_acmtf(i+n_samples,:)] = ...
        comp_err_facs(A{2}{i}, [Facmtf{i+n_samples}{1}.U;...
        Facmtf{i+n_samples}{2}.U(1)], X{2}{i}.orig, modes, Facmtf{i+n_samples});
%     [err_acmtf_sc(i,:), corr_acmtf_sc(i,:)] = comp_err_facs(A{1}{i}(mds),...
%         [Facmtf_sc{i}{1}.U; Facmtf_sc{i}{2}.U], X{1}{i}.orig, {[1:3],[4:length(mds)]});
%     [err_acmtf_sc(i+n_samples,:), corr_acmtf_sc(i+n_samples,:)] = ...
%         comp_err_facs(A{2}{i}(mds), [Facmtf_sc{i+n_samples}{1}.U;...
%         Facmtf_sc{i+n_samples}{2}.U], X{2}{i}.orig, {[1:3],[4:length(mds)]});
end

for i = 0:15
    w_list(i+1,:) = [floor(i/8), floor(mod(i,8)/4), floor(mod(i,4)/2), mod(i,2)];
end
    
F(1,:) = Fcp;
F(2,:) = Fpca;
F(3,:) = Fcmtf;
F(4,:) = Facmtf;
results = test_wcval(A, F, w_list, K);

for i_folds=1:K
    results(i_folds).orig.err = err_orig; results(i_folds).orig.corr = corr_orig;
    results(i_folds).orig.fms = fms_orig;
    results(i_folds).cp.err = err_cp; results(i_folds).cp.corr = corr_cp;
    results(i_folds).cp.fms = fms_cp;
    results(i_folds).pca.err = err_pca; results(i_folds).pca.corr = corr_pca;
    results(i_folds).pca.fms = fms_pca;
    results(i_folds).cmtf.err = err_cmtf; results(i_folds).cmtf.corr = corr_cmtf;
    results(i_folds).cmtf.fms = fms_cmtf;
    results(i_folds).acmtf.err = err_acmtf; results(i_folds).acmtf.corr = corr_acmtf;
    results(i_folds).acmtf.fms = fms_acmtf;
end


end

function [err, corrs, fms] = comp_err_facs(X, F, Y, modes, varargin)

if length(X)~=length(F) || ~iscell(X) || ~iscell(F)
    error('Wrong input type!')
end
N = length(X);
corrs = zeros(1,N);
for n=1:N
    C = corr(X{n}, F{n});
    corrs(n) = mean(max(abs(C)));
end

P = length(modes);
fms = zeros(1,length(unique(cell2mat(modes))));
for p=1:P
    for n=1:length(modes{p})
        fms(modes{p}(n)) = score(ktensor(X(modes{p}(n))), ktensor(F(modes{p}(n))));
    end
end

err = zeros(1,length(modes));
if nargin==4
    for p=1:P
        found = ktensor(F(modes{p}));
        err(p) = norm(Y{p}-full(found))/norm(Y{p});
    end
else
    T = varargin{1};
    for p=1:P
        found = full(T{p});
        err(p) = norm(Y{p}-full(found))/norm(Y{p});
    end
end
end

% 
% function results = cpld_class(X, A, K)
% 
% n_samples = length(X{1});
% n_test = round(n_samples./K);
% n_train = n_samples-n_test;
% R = size(A{1}{1}{1},2);
% modes = X{1}{1}.modes;
% mds = cell2mat(X{1}{1}.modes);
% P = length(modes);
% beta_cp = .001; beta_pca = .001;
% alpha = 0.1;
% 
% options = ncg('defaults');
% options.Display ='off';
% options.MaxFuncEvals = 100000;
% options.MaxIters     = 500;
% options.StopTol      = 1e-7;
% options.RelFuncTol   = 1e-7;
% options.TraceFunc    = true;
% 
% err_orig = zeros(n_samples*2, P); corr_orig = zeros(n_samples*2, length(mds)-1);
% err_cp = zeros(n_samples*2, 1); err_pca = zeros(n_samples*2, 1);
% err_cmtf = zeros(n_samples*2, P); err_acmtf = zeros(n_samples*2, P);
% err_acmtf_sc = zeros(n_samples*2, P);
% corr_cp = zeros(n_samples*2, length(modes{1})); corr_pca = zeros(n_samples*2, length(modes{2}));
% corr_cmtf = zeros(n_samples*2, length(mds)-1); 
% corr_acmtf = zeros(n_samples*2, length(mds)-1);
% corr_acmtf_sc = zeros(n_samples*2, length(mds));
% fms_cp = zeros(n_samples*2, 1); fms_pca = zeros(n_samples*2,1);
% fms_cmtf = zeros(n_samples*2, P); fms_acmtf = zeros(n_samples*2, P);
% fms_acmtf_sc = zeros(n_samples*2, P); fms_orig = zeros(n_samples*2, P);
% 
% Fcp = cell(1, 2*n_samples);
% Fpca = Fcp; Fcmtf = Fpca; Facmtf = Fcmtf; Facmtf_sc = Facmtf;
% for i=1:n_samples
%     %% Extract Factors
% %     Fcp{i} = cp_wopt(X{1}{i}.object{1}, X{1}{i}.miss{1}, R);
% %     Fpca{i} = cp_wopt(X{1}{i}.object{2}, X{1}{i}.miss{2}, R);
%     Fcp{i} = cpd(double(X{1}{i}.object{1}), R);%, 'alg_options', options);
%     Fpca{i} = cpd(double(X{1}{i}.object{2}), R);%, 'alg_options', options);
%     Fcmtf{i} = extract_w_CMTF(X{1}{i}, [], R, 'modes', modes);
%     Facmtf{i} = extract_w_ACMTF(X{1}{i}, [], R, 'modes', modes,'beta_cp',...
%         beta_cp, 'beta_pca', beta_pca);
% %     Facmtf_sc{i} = extract_wsc_ACMTF(X{1}{i},[], R, 'modes', modes,'beta_cp',...
% %         beta_cp, 'beta_pca', beta_pca, 'alpha', alpha);
% %     Fcp{i+n_samples} = cp_wopt(X{2}{i}.object{1}, X{2}{i}.miss{1}, R);
% %     Fpca{i+n_samples} = cp_wopt(X{2}{i}.object{2}, X{2}{i}.miss{2}, R);
%     Fcp{i+n_samples} = cpd(double(X{2}{i}.object{1}), R);%, 'alg_options', options);
%     Fpca{i+n_samples} = cpd(double(X{2}{i}.object{2}), R);%, 'alg_options', options);
%     Fcmtf{i+n_samples} = extract_w_CMTF(X{2}{i}, [], R, 'modes', modes);
%     Facmtf{i+n_samples} = extract_w_ACMTF(X{2}{i}, [], R, 'modes', modes,...
%         'beta_cp', beta_cp, 'beta_pca', beta_pca);
% %     Facmtf_sc{i+n_samples} = extract_wsc_ACMTF(X{2}{i}, [], R, 'modes',...
% %         modes,'beta_cp', beta_cp, 'beta_pca', beta_pca, 'alpha', alpha);
%     
%     %% Compute the decomposition quality.
%     [err_orig(i,:), corr_orig(i,:), fms_orig(i,:)] = ...
%         comp_err_facs(A{1}{i}, A{2}{i}, X{1}{i}.orig, modes);
%     err_orig(i+n_samples,:) = err_orig(i,:);
%     corr_orig(i+n_samples,:) = corr_orig(i,:);
%     [err_cp(i,:), corr_cp(i,:), fms_cp(i)] = comp_err_facs(A{1}{i}(modes{1}), Fcp{i},...
%         X{1}{i}.orig(1), {1:length(modes{1})});
%     [err_cp(i+n_samples,:), corr_cp(i+n_samples,:), fms_cp(i+n_samples)] = ...
%         comp_err_facs(A{2}{i}(modes{1}), Fcp{i+n_samples}, X{2}{i}.orig(1),...
%         {1:length(modes{1})});
%     [err_pca(i,:), corr_pca(i,:), fms_pca(i)] = comp_err_facs(A{1}{i}(modes{2}), Fpca{i},...
%         X{1}{i}.orig(2), {1:length(modes{2})});
%     [err_pca(i+n_samples,:), corr_pca(i+n_samples,:), fms_pca(i+n_samples)] = ...
%         comp_err_facs(A{2}{i}(modes{2}), Fpca{i+n_samples}, X{2}{i}.orig(2),...
%         {1:length(modes{2})});
%     [err_cmtf(i,:), corr_cmtf(i,:), fms_cmtf(i,:)] = comp_err_facs(A{1}{i},...
%         [Fcmtf{i}{1}.U; Fcmtf{i}{2}.U(1)],X{1}{i}.orig, modes, Fcmtf{i});
%     [err_cmtf(i+n_samples,:), corr_cmtf(i+n_samples,:), fms_cmtf(i+n_samples,:)] = ...
%         comp_err_facs(A{2}{i}, [Fcmtf{i+n_samples}{1}.U;...
%         Fcmtf{i+n_samples}{2}.U(1)], X{2}{i}.orig, modes, Fcmtf{i+n_samples});
%     [err_acmtf(i,:), corr_acmtf(i,:), fms_acmtf(i,:)] = comp_err_facs(A{1}{i},...
%         [Facmtf{i}{1}.U; Facmtf{i}{2}.U(1)], X{1}{i}.orig, modes, Facmtf{i});
%     [err_acmtf(i+n_samples,:), corr_acmtf(i+n_samples,:), fms_acmtf(i+n_samples,:)] = ...
%         comp_err_facs(A{2}{i}, [Facmtf{i+n_samples}{1}.U;...
%         Facmtf{i+n_samples}{2}.U(1)], X{2}{i}.orig, modes, Facmtf{i+n_samples});
% %     [err_acmtf_sc(i,:), corr_acmtf_sc(i,:)] = comp_err_facs(A{1}{i}(mds),...
% %         [Facmtf_sc{i}{1}.U; Facmtf_sc{i}{2}.U], X{1}{i}.orig, {[1:3],[4:length(mds)]});
% %     [err_acmtf_sc(i+n_samples,:), corr_acmtf_sc(i+n_samples,:)] = ...
% %         comp_err_facs(A{2}{i}(mds), [Facmtf_sc{i+n_samples}{1}.U;...
% %         Facmtf_sc{i+n_samples}{2}.U], X{2}{i}.orig, {[1:3],[4:length(mds)]});
% end
% 
% for i = 0:15
%     w_list(i+1,:) = [floor(i/8), floor(mod(i,8)/4), floor(mod(i,4)/2), mod(i,2)];
% end
%     
% F(1,:) = Fcp;
% F(2,:) = Fpca;
% F(3,:) = Fcmtf;
% F(4,:) = Facmtf;
% results = test_wcval(A, F, w_list, K);
% 
% for i_folds=1:K
%     results(i_folds).orig.err = err_orig; results(i_folds).orig.corr = corr_orig;
%     results(i_folds).orig.fms = fms_orig;
%     results(i_folds).cp.err = err_cp; results(i_folds).cp.corr = corr_cp;
%     results(i_folds).cp.fms = fms_cp;
%     results(i_folds).pca.err = err_pca; results(i_folds).pca.corr = corr_pca;
%     results(i_folds).pca.fms = fms_pca;
%     results(i_folds).cmtf.err = err_cmtf; results(i_folds).cmtf.corr = corr_cmtf;
%     results(i_folds).cmtf.fms = fms_cmtf;
%     results(i_folds).acmtf.err = err_acmtf; results(i_folds).acmtf.corr = corr_acmtf;
%     results(i_folds).acmtf.fms = fms_acmtf;
% end
% 
% 
% end
% 
% function [err, corrs, fms] = comp_err_facs(X, F, Y, modes, varargin)
% 
% if length(X)~=length(F) || ~iscell(X) || ~iscell(F)
%     error('Wrong input type!')
% end
% N = length(X);
% corrs = zeros(1,N);
% for n=1:N
%     C = corr(X{n}, F{n});
%     corrs(n) = mean(max(abs(C)));
% end
% 
% P = length(modes);
% fms = zeros(1,P);
% for p=1:P
%     fms(p) = score(ktensor(X(modes{p})), ktensor(F(modes{p})));
% end
% 
% err = zeros(1,length(modes));
% if nargin==4
%     for p=1:P
%         found = ktensor(F(modes{p}));
%         err(p) = norm(Y{p}-full(found))/norm(Y{p});
%     end
% else
%     T = varargin{1};
%     for p=1:P
%         found = full(T{p});
%         err(p) = norm(Y{p}-full(found))/norm(Y{p});
%     end
% end
% end