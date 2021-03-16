
clear fMRI_stacked eeg_stacked classLoss

l = size(oddball_fMRI{1,1},2);
sz = size(oddball_fMRI{1,1});
fMRI_stacked(:,1:l,:) = cat(3,oddball_fMRI{1,1}, standard_fMRI{1,1});
fMRI_stacked(:,l+1:2*l,:) = cat(3,oddball_fMRI{1,2}, standard_fMRI{1,2});
fMRI_stacked(:,2*l+1:3*l,:) = cat(3,oddball_fMRI{1,3}, standard_fMRI{1,3});
eeg_stacked(:,1:l,:) = cat(3,oddball_eeg{1,1}, standard_eeg{1,1});
eeg_stacked(:,l+1:2*l,:) = cat(3,oddball_eeg{1,2}, standard_eeg{1,2});
eeg_stacked(:,2*l+1:3*l,:) = cat(3,oddball_eeg{1,3}, standard_eeg{1,3});


fMRI_stacked = reshape(fMRI_stacked, [sz(1),l*6]);
eeg_stacked = reshape(eeg_stacked, [37,2000,l*6]);
labels = [ones(l*3,1); -ones(l*3,1)];

ranks = [15];
w = [10.^[-3:-1:-5]];
for i=1:length(ranks)
    for j=1:length(w)
        [F, out_SDF] = extractFactors_eegfMRI(eeg_stacked, fMRI_stacked, ranks(i), w(j), 1);
        coupled_features = F.factors.U3;
        
        SVM_Model = fitcsvm(coupled_features, labels);
        CVSVMModel = crossval(SVM_Model);
        classLoss(1,i,j,:) = kfoldLoss(CVSVMModel,'mode','individual');
        
        [Facmtf, out_acmtf] = extract_w_ACMTF(eeg_stacked, fMRI_stacked, ranks(i), w(j), 1);
        coupled_features = Facmtf{1}.u{3};
        
        SVM_Model = fitcsvm(coupled_features, labels);
        CVSVMModel = crossval(SVM_Model);
        classLoss(2,i,j,:) = kfoldLoss(CVSVMModel,'mode','individual');
    end
end
1-mean(classLoss,4)
std(classLoss,[],4)