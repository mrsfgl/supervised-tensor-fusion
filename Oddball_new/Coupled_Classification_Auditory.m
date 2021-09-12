% Perform coupled tensor decomposition and C-STM classification using
% Auditory EEG and fMRI data

% Auditory stimulus
clear; clc;
%%
EEG_dir = 'D:\Research\EEG-fMRI\Auditory_EEG';
fMRI_dir = 'D:\Research\EEG-fMRI\Auditory_fMRI';


fMRI_list = dir(fMRI_dir);

behav_dir = 'D:\Research\EEG-fMRI\Auditory_label';

n_subjects = 17;
oddball_fMRI = cell(n_subjects, 3);
standard_fMRI = cell(n_subjects, 3);
oddball_eeg = cell(n_subjects, 3);
standard_eeg = cell(n_subjects, 3);
n_trials = 15;
for i = 1:n_subjects
    if i == 4   
%         Skip indivudial 4 since one of its fMRI cannot be read
        continue
    end
    
    % For each subject extract fMRI and EEG for 3 auditory task data. Apply
    % necessary reshaping operations such that factors match in size.
    imfMRI = zeros(64*64*32, 170, 3);
    EEG = zeros(37, 340000, 3);
    for j=1:3
        if i == 4 && j == 2
            continue
        end
        fMRI_name = join(['sub-', num2str(i, '%02d'), '_task-auditoryoddballwithbuttonresponsetotargetstimuli_run-',...
            num2str(j, '%02d'), '_bold.nii.gz'], '');
        I = niftiread(join([fMRI_dir, filesep, fMRI_name]));

        tsv_name = join([behav_dir, filesep, 'sub-', num2str(i, '%02d'), '_task-auditoryoddballwithbuttonresponsetotargetstimuli_run-',...
            num2str(j, '%02d'), '_events.tsv'], '');
        imfMRI(:, :, j) = cast(reshape(I,[],170),'single');
        [oddball_fMRI{i, j}, standard_fMRI{i, j}] = trial_sep_eeg_fMRI(imfMRI(:, :, j), tsv_name, n_trials);
        
        EEG_name = join(['Sub', num2str(i, '%02d'), '_Auditory_run', num2str(j, '%02d'), '.mat'], '');
        load(join([EEG_dir, filesep, EEG_name], ''));
        EEG(:, :, j) = cast(data_reref, 'single');
        [oddball_eeg{i, j}, standard_eeg{i, j}] = trial_sep_eeg_fMRI(reshape(EEG(:, :, j), [], 170), tsv_name, n_trials);
        
    end
    
end





%% Reshape trail-separated data into tensors

Tensor_EEG_Std = [];
Tensor_fMRI_odd = [];
Tensor_EEG_odd = [];
Tensor_fMRI_Std = [];
for j = 1 : 3
   tmp_eeg_odd = [];
   tmp_eeg_std = [];
   tmp_fMRI_odd = [];
   tmp_fMRI_std = [];
   for i = 1 : n_subjects
       if i == 4
           continue
       end
       
       tmp_eeg_odd = cat(4, tmp_eeg_odd, reshape(oddball_eeg{i, j}, 37, 2000, []));
       tmp_eeg_std = cat(4, tmp_eeg_std, reshape(standard_eeg{i, j}, 37, 2000, []));
       tmp_fMRI_odd = cat(3, tmp_fMRI_odd, oddball_fMRI{i, j});
       tmp_fMRI_std = cat(3, tmp_fMRI_std, standard_fMRI{i, j});  
   end 
   Tensor_EEG_Std = cat(4, Tensor_EEG_Std, permute(tmp_eeg_std, [1, 2, 4, 3]));
   Tensor_EEG_odd = cat(4, Tensor_EEG_odd, permute(tmp_eeg_odd, [1, 2, 4, 3]));
   Tensor_fMRI_Std = cat(3, Tensor_fMRI_Std, permute(tmp_fMRI_std, [1, 3, 2]));
   Tensor_fMRI_odd = cat(3, Tensor_fMRI_odd, permute(tmp_fMRI_odd, [1, 3, 2]));
end



%% perform Coupled Decomposition
sz = size(Tensor_EEG_Std);
D_data = cell(1, 2 * sz(4));
y_label = zeros(1, 2 * sz(4));
for i = 1 : sz(4)
   eeg = squeeze(Tensor_EEG_Std(:, :, :, i));
   fMRI = squeeze(Tensor_fMRI_Std(:, :, i));
   [g, ~] = ExtractFactors_EEGfMRI(eeg, fMRI, 5, 1, 1);
   D_data{i} = g.variables;
   y_label(i) = -1;
end



%%
for j = 1 : sz(4)
   eeg = squeeze(Tensor_EEG_odd(:, :, :, j));
   fMRI = squeeze(Tensor_fMRI_odd(:, :, j));
   [g, ~] = ExtractFactors_EEGfMRI(eeg, fMRI, 5, 1, 1);
   D_data{j + sz(4)} = g.variables;
   y_label(j + sz(4)) = 1;
end



%%
D_new = cell(1, 90);
for i = 1 : 90
    D_new{i} = D_data{i};
end




%% Perform Classification: Use K-Folds Corss Validation to evulate model performance

load('matlab.mat');
%%

y_label = [-1.* ones(1, 45), ones(1, 45)];

K_folds = crossvalind('Kfold', y_label, 9);




%% Validation procedure

accuracy = zeros(1, 9);
precision = zeros(1, 9);
recall = zeros(1, 9);
specificity = zeros(1, 9);

for i = 1 : 9
    train_set = cell(1, 80);
    test_set = cell(1, 10);
    y_train = zeros(1, 80);
    y_test = zeros(1, 10);
    k = 1;
    l = 1;
    for j = 1 : 90
        if K_folds(j) == i
            test_set{k} = D_new{j};
            if j <= 45
                y_test(k) = -1; 
            else
                y_test(k) = 1;
            end
            k = k + 1;
        else
            train_set{l} = D_new{j};
            if j <= 45
                y_train(l) = -1;
            else
                y_train(l) = 1;
            end
            l = l + 1;
        end
    end 
    [alpha, b] = Coupled_EEG_fMRI_STM(train_set, y_train, 1 / length(train_set), [0.1, 0.01], [0.5], [0.00001]);
    y_predict = Coupled_EEG_fMRI_STM_Predict(test_set, train_set, y_train, alpha, b, [0.1, 0.01], [0.5], [0.00001]);
    perf = classperf((y_test == 1), (y_predict == 1));
    accuracy(i) = perf.CorrectRate;
    precision(i) = perf.PositivePredictiveValue;
    recall(i) = perf.Sensitivity;
    specificity(i) = perf.Specificity;
end























