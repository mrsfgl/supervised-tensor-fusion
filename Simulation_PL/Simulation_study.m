
% Perform coupled tensor decomposition and C-STM classification using
% Visual EEG and fMRI data

% fMRI data are contrast coefficients
clear; clc;

addpath("D:\Research\EEG-fMRI\EEG_fMRI_Code\poblano_toolbox-main");
addpath("D:\Research\EEG-fMRI\EEG_fMRI_Code\CMTF_Toolbox_v1_1");
addpath("D:\Research\EEG-fMRI\EEG_fMRI_Code\Simulation Study\CoupledSTM");
addpath("D:\Research\EEG-fMRI\EEG_fMRI_Code\CoupledDecomposition");
addpath("D:\Research\EEG-fMRI\EEG_fMRI_Code\Tensorlabs");
addpath("D:\Research\EEG-fMRI\EEG_fMRI_Code\tensor_toolbox-master")

%% load data
for t = 1 : 5

load(sprintf('Simulation_Noise_%d.mat', t));


% load(sprintf('New_noNoise_%d.mat', t));


%% perform Coupled Decomposition for training data
n = 50;
D_data = cell(1, 2 * n);

for i = 1 : n
   eeg = X{1}{i}{1};
   fMRI = X{1}{i}{2};
   [g, ~] = extract_w_CMTF(eeg, fMRI, 3);
   D_data{i} = g;
%     df = double(X{1}{i}{2});
%     g = cpd(df, 3);
%     D_data{i} = g;
end


for j = 1 : n
   eeg = X{2}{j}{1};
   fMRI = X{2}{j}{2};
   [g, ~] = extract_w_CMTF(eeg, fMRI, 3);
   D_data{j + n} = g;
%     df = double(X{2}{j}{2});
%     g = cpd(df, 3);
%     D_data{j + n} = g;
end



%% perform Coupled Decomposition for test data
y = [ones(1, n), -1.*ones(1, n)];

accuracy = [];
precision = [];
recall = [];
specificity = [];

for i = 1 : 50
    s = cvpartition(ones(size(y)), 'HoldOut', 0.2);
    Xtrain = D_data(s.training);
    Xtest = D_data(s.test);
    ytrain = y(s.training);
    ytest = y(s.test);
    [alpha, b] = CMTF_STM(Xtrain, ytrain, 10, [1, 1], [1], [1], 'QP');
    ypredict = CMTF_STM_Predict(Xtest, Xtrain, ytrain, alpha, b, [1, 1], [1], [1]);
    perf = classperf((ytest == 1), (ypredict == 1));
    accuracy(end + 1) = perf.CorrectRate;
    precision(end + 1) = perf.PositivePredictiveValue;
    recall(end + 1) = perf.Sensitivity;
    specificity(end + 1) = perf.Specificity;
end

K = table(accuracy', precision', recall', specificity');
writetable(K, sprintf('Result_Noise_%d_CMTF.csv', t));
end
% 