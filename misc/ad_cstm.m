
% load('rec_errs_facs_ad_acmtf_alpha_1e-3_rlist.mat')
load('class_order.mat')

F = cell(1,size(Fcmtf,2));
results = F;
for j = 1:size(Fcmtf,2)
    F{j}(1,:) = Fcp_1(ind_true,j);
    F{j}(2,:) = Fcp_2(ind_true,j);
    F{j}(3,:) = Fcmtf(ind_true,j);
    F{j}(4,:) = Facmtf(ind_true,j);

    w_list = [];
    folds = 10;
    results{j} = test_wcval([],F{j},w_list,folds);
end