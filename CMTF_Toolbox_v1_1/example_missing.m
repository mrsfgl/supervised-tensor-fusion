%% ACMTF
data = tester_acmtf_missing;

trueval = data.Xorig{1}(find(data.W{1}==0));
Z       = full(data.Zhat{1});
estval  = Z(find(data.W{1}==0));
plot(trueval,estval,'*')
err     = norm(estval - trueval)/length(estval);

data = tester_acmtf_missing('flag_sparse',[1 1]);
%% CMTF
data = tester_cmtf_missing;
trueval = data.Xorig{1}(find(data.W{1}==0));
Z       = full(ktensor(data.Fac(1:3)));
estval  = Z(find(data.W{1}==0));
plot(trueval,estval,'*')
err     = norm(estval - trueval)/length(estval);


data = tester_cmtf_missing('flag_sparse',[1 1]);