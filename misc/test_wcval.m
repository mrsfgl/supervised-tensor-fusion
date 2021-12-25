function [results] = test_wcval(A, F, w_list, K)

n_samples = size(F,2)/2;
n_val = round(n_samples/K);
n_train = n_samples-n_val;

% n_cval = size(w_list,1);
% cval_train = [1:n_train-n_val,n_samples+(1:n_train-n_val)];
% cval_val = [n_train+1:n_samples, n_samples+(n_train+1:n_samples)];
% train = F(4,cval_train);
% val = F(4,cval_val);
% for i_cval = 1:n_cval
%     cval_results(i_cval) = classify_factors(train, val, w_list(i_cval,:));
%     acc(i_cval) = cval_results(i_cval).accuracy;
% end
F = F(:,[1:n_train, n_samples+(1:n_train)]);
if isempty(A)
    A{1} = [];
    A{2} = [];
else
    A{1} = A{1}(:,1:n_train);
    A{2} = A{2}(:,1:n_train);
end
% [~,i_wgt] = max(acc);
% w = w_list(i_wgt, :);
w = [.25,.25,.25,.25];

n_samples = n_train;

rng(123)
K_folds = crossvalind('Kfold', [ones(1,n_samples),-ones(1,n_samples)],K);
for i_folds=1:K
    [train_1, test_1] = tr_test_split(K_folds, i_folds, 0, A{1}, F);
    [train_2, test_2] = tr_test_split(K_folds, i_folds, n_samples, A{2}, F);
    train = cat(2,train_1,train_2);
    test = cat(2,test_1,test_2);
    
    if ~isempty(A{1})
        results(i_folds).orig = classify_factors(train(1,:), test(1,:), w);
    end
    results(i_folds).cp = classify_factors(train(2,:), test(2,:), w);
    results(i_folds).pca = classify_factors(train(3,:), test(3,:), w);
    results(i_folds).cmtf = classify_factors(train(4,:), test(4,:), w);
    results(i_folds).acmtf = classify_factors(train(5,:), test(5,:), w);
%     results(i_folds).acmtf_sc = classify_factors(train_acmtf_sc, test_acmtf_sc);
%     results(i_folds).acmtf_sc.err = err_acmtf_sc; results(i_folds).acmtf_sc.corr = corr_acmtf_sc;
end
end

function res = classify_factors(train, test, weight)

wght_U = weight(1:2);
wght_C = weight(3);
wght_V = weight(4:end);
l_tr = length(train);
l_tst = length(test);
y_train = [ones(1,l_tr/2), -ones(1,l_tr/2)];
y_test = [ones(1,l_tst/2), -ones(1,l_tst/2)];

[alpha, b] = coupled_stm(train, y_train, 10, wght_U, wght_C, wght_V);
y_predict = coupled_stm_predict(test, train, y_train, alpha, b, wght_U, wght_C, wght_V);
perf = classperf((y_test == 1), (y_predict == 1));
res.accuracy = perf.CorrectRate;
res.precision = perf.PositivePredictiveValue;
res.recall = perf.Sensitivity;
res.specificity = perf.Specificity;
end

function [train, test] = tr_test_split(K_folds, i_folds, offset, A, F)

n_samples = size(F,2)/2;
l = 1;
k = 1;
for i=1:n_samples
    if K_folds(i)==i_folds
        if ~isempty(A)
            test{1,k} = ktensor(A{i});
        end
        test{2,k} = F{1,i+offset(1)};
        test{3,k} = F{2,i+offset(1)};
        test{4,k} = F{3,i+offset(1)};
        test{5,k} = F{4,i+offset(1)};
%             test_acmtf_sc{k} = Facmtf_sc{i+n_samples};
        k = k+1;
    else
        if ~isempty(A)
            train{1,l} = ktensor(A{i});
        end
        train{2,l} = F{1,i+offset(1)};
        train{3,l} = F{2,i+offset(1)};
        train{4,l} = F{3,i+offset(1)};
        train{5,l} = F{4,i+offset(1)};
%             train_acmtf_sc{l} = Facmtf_sc{i+n_samples};
        l = l+1;
    end
end
end