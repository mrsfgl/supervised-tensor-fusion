function results = cpld_class(X, A, K)

n_samples = length(X{1});
n_train = round(n_samples*.9);
n_test = n_samples-n_train;
R = size(A{1}{1}{1},2);
modes = X{1}{1}.modes;
mds = cell2mat(X{1}{1}.modes);
P = length(modes);

train_orig = cell(1,n_train); test_orig = cell(1,n_test);
train_cp = cell(1,n_train); test_cp = cell(1,n_test);
train_pca = cell(1,n_train); test_pca = cell(1,n_test);
train_cmtf = cell(1,n_train); test_cmtf = cell(1,n_test);
train_acmtf = cell(1,n_train); test_acmtf = cell(1,n_test);
train_acmtf_sc = cell(1,n_train); test_acmtf_sc = cell(1,n_test);
err_orig = zeros(n_samples, P); corr_orig = zeros(n_samples, 4);
err_cp = zeros(n_samples, 1); err_pca = zeros(n_samples, 1);
err_cmtf = zeros(n_samples, P); err_acmtf = zeros(n_samples, P);
err_acmtf_sc = zeros(n_samples, P);
corr_cp = zeros(n_samples, 3); corr_pca = zeros(n_samples, 2);
corr_cmtf = zeros(n_samples, 4); corr_acmtf = zeros(n_samples, 4);
corr_acmtf_sc = zeros(n_samples, 5);

K_folds = crossvalind('Kfold', [ones(1,n_samples),-ones(1,n_samples)],K);
Fcp = cell(1, 2*n_samples);
Fpca = Fcp; Fcmtf = Fpca; Facmtf = Fcmtf; Facmtf_sc = Facmtf;
for i=1:n_samples
    %% Extract Factors
    Fcp{i} = cp_wopt(X{1}{i}.object{1}, X{1}{i}.miss{1}, R);
    Fpca{i} = cp_wopt(X{1}{i}.object{2}, X{1}{i}.miss{2}, R);
    Fcmtf{i} = extract_w_CMTF(X{1}{i}, [], R, 'modes', modes);
    Facmtf{i} = extract_w_ACMTF(X{1}{i}, [], R, 'modes', modes);
    Facmtf_sc{i} = extract_wsc_ACMTF(X{1}{i},[], R, 'modes', modes);
    Fcp{i+n_samples} = cp_wopt(X{2}{i}.object{1}, X{2}{i}.miss{1}, R);
    Fpca{i+n_samples} = cp_wopt(X{2}{i}.object{2}, X{2}{i}.miss{2}, R);
    Fcmtf{i+n_samples} = extract_w_CMTF(X{2}{i}, [], R, 'modes', modes);
    Facmtf{i+n_samples} = extract_w_ACMTF(X{2}{i}, [], R, 'modes', modes);
    Facmtf_sc{i+n_samples} = extract_wsc_ACMTF(X{2}{i}, [], R, 'modes', modes);
    
    %% Compute the decomposition quality.
    [err_orig(i,:), corr_orig(i,:)] = ...
        comp_err_facs(A{1}{i}, A{2}{i}, X{1}{i}.orig, modes);
    err_orig(i+n_samples,:) = err_orig(i,:);
    corr_orig(i+n_samples,:) = corr_orig(i,:);
    [err_cp(i,:), corr_cp(i,:)] = comp_err_facs(A{1}{i}(1:3), Fcp{i}.U,...
        X{1}{i}.orig(1), {1:3});
    [err_pca(i,:), corr_pca(i,:)] = comp_err_facs(A{1}{i}([1,4]), Fpca{i}.U,...
        X{1}{i}.orig(2), {1:2});
    [err_cmtf(i,:), corr_cmtf(i,:)] = comp_err_facs(A{1}{i}, Fcmtf{i}.U,...
        X{1}{i}.orig, modes);
    [err_acmtf(i,:), corr_acmtf(i,:)] = comp_err_facs(A{1}{i},...
        [Facmtf{i}{1}.U; Facmtf{i}{2}.U(2)], X{1}{i}.orig, modes);
    [err_acmtf_sc(i,:), corr_acmtf_sc(i,:)] = comp_err_facs(A{1}{i}(mds),...
        [Facmtf_sc{i}{1}.U; Facmtf_sc{i}{2}.U], X{1}{i}.orig, {[1:3],[4,5]});
    [err_cp(i+n_samples,:), corr_cp(i+n_samples,:)] = ...
        comp_err_facs(A{2}{i}(1:3), Fcp{i+n_samples}.U, X{2}{i}.orig(1), {1:3});
    [err_pca(i+n_samples,:), corr_pca(i+n_samples,:)] = ...
        comp_err_facs(A{2}{i}([1,4]), Fpca{i+n_samples}.U, X{2}{i}.orig(2), {1:2});
    [err_cmtf(i+n_samples,:), corr_cmtf(i+n_samples,:)] = ...
        comp_err_facs(A{2}{i}, Fcmtf{i+n_samples}.U, X{2}{i}.orig, modes);
    [err_acmtf(i+n_samples,:), corr_acmtf(i+n_samples,:)] = ...
        comp_err_facs(A{2}{i}, [Facmtf{i+n_samples}{1}.U;...
        Facmtf{i+n_samples}{2}.U(2)], X{2}{i}.orig, modes);
    [err_acmtf_sc(i+n_samples,:), corr_acmtf_sc(i+n_samples,:)] = ...
        comp_err_facs(A{2}{i}(mds), [Facmtf_sc{i+n_samples}{1}.U;...
        Facmtf_sc{i+n_samples}{2}.U], X{2}{i}.orig, {[1:3],[4,5]});
end

for i_folds=1:K
    k=1;
    l=1;
    for i=1:n_samples
        if K_folds(i)==i_folds
            test_orig{k} = ktensor(A{1}{i});
            test_cp{k} = Fcp{i};
            test_pca{k} = Fpca{i};
            test_cmtf{k} = Fcmtf{i};
            test_acmtf{k} = Facmtf{i};
            test_acmtf_sc{k} = Facmtf_sc{i};
            k = k+1;
        else
            train_orig{l} = ktensor(A{1}{i});
            train_cp{l} = Fcp{i};
            train_pca{l} = Fpca{i};
            train_cmtf{l} = Fcmtf{i};
            train_acmtf{l} = Facmtf{i};
            train_acmtf_sc{l} = Facmtf_sc{i};
            l = l+1;
        end
    end
    for i=1:n_samples
        if K_folds(i)==i_folds
            test_orig{k} = ktensor(A{2}{i});
            test_cp{k} = Fcp{i+n_samples};
            test_pca{k} = Fpca{i+n_samples};
            test_cmtf{k} = Fcmtf{i+n_samples};
            test_acmtf{k} = Facmtf{i+n_samples};
            test_acmtf_sc{k} = Facmtf_sc{i+n_samples};
            k = k+1;
        else
            train_orig{l} = ktensor(A{2}{i});
            train_cp{l} = Fcp{i+n_samples};
            train_pca{l} = Fpca{i+n_samples};
            train_cmtf{l} = Fcmtf{i+n_samples};
            train_acmtf{l} = Facmtf{i+n_samples};
            train_acmtf_sc{l} = Facmtf_sc{i+n_samples};
            l = l+1;
        end
    end
    results(i_folds).orig = classify_factors(train_orig, test_orig);
    results(i_folds).orig.err = err_orig; results(i_folds).orig.corr = corr_orig;
    results(i_folds).cp = classify_factors(train_cp, test_cp);
    results(i_folds).cp.err = err_cp; results(i_folds).cp.corr = corr_cp;
    results(i_folds).pca = classify_factors(train_pca, test_pca);
    results(i_folds).pca.err = err_pca; results(i_folds).pca.corr = corr_pca;
    results(i_folds).cmtf = classify_factors(train_cmtf, test_cmtf);
    results(i_folds).cmtf.err = err_cmtf; results(i_folds).cmtf.corr = corr_cmtf;
    results(i_folds).acmtf = classify_factors(train_acmtf, test_acmtf);
    results(i_folds).acmtf.err = err_acmtf; results(i_folds).acmtf.corr = corr_acmtf;
    results(i_folds).acmtf_sc = classify_factors(train_acmtf_sc, test_acmtf_sc);
    results(i_folds).acmtf_sc.err = err_acmtf_sc; results(i_folds).acmtf_sc.corr = corr_acmtf_sc;
end


end

function res = classify_factors(train, test)

l_tr = length(train);
l_tst = length(test);
y_train = [ones(1,l_tr/2), -ones(1,l_tr/2)];
y_test = [ones(1,l_tst/2), -ones(1,l_tst/2)];

[alpha, b] = Coupled_EEG_fMRI_STM(train, y_train, 1/l_tr, [0.1, 0.01], [0.5], [0.00001]);
y_predict = Coupled_EEG_fMRI_STM_Predict(test, train, y_train, alpha, b, [0.1, 0.01], [0.5], [0.00001]);
perf = classperf((y_test == 1), (y_predict == 1));
res.accuracy = perf.CorrectRate;
res.precision = perf.PositivePredictiveValue;
res.recall = perf.Sensitivity;
res.specificity = perf.Specificity;
end

function [err, corrs] = comp_err_facs(X, F, Y, modes)

if length(X)~=length(F) || ~iscell(X) || ~iscell(F)
    error('Wrong input type!')
end
N = length(X);
corrs = zeros(1,N);
for n=1:N
    C = corr(X{n}, F{n});
    corrs(n) = mean(max(abs(C)));
end
err = zeros(1,length(modes));
for p=1:length(modes)
    found = ktensor(F(modes{p}));
    err(p) = norm(Y{p}-full(found))/norm(Y{p});
end
end