
clear
clc
addpath(genpath('M:/Documents/MATLAB/tboxunzipped/tensorlab'))

% Specifying data path should be enough to read and process all data. Check
% if all subjects have T1 images and 3 auditory oddball fMRI.
if isunix ==1
    path = '/egr/research/sigimprg/Emre/databases/EEG-fMRI/EEG_FMRI';
    list_subjects = ls(path);
    list_subjects = split(list_subjects);
    n_subjects = size(list_subjects,1)-1;
    for i =1:n_subjects
        temp(i,:) = list_subjects{i};
    end
    list_subjects = temp;
else
    path = '\\cifs.egr.msu.edu\research\sigimprg\Emre\databases\EEG-fMRI\EEG_FMRI';
    list_subjects = ls(path);
    list_subjects = list_subjects(3:end,:);
    n_subjects = size(list_subjects,1);
end


oddball_fMRI = cell(n_subjects, 3);
standard_fMRI = cell(n_subjects, 3);
oddball_eeg = cell(n_subjects, 3);
standard_eeg = cell(n_subjects, 3);
for i = 1:n_subjects
    % For each subject extract fMRI and EEG for 3 auditory task data. Apply
    % necessary reshaping operations such that factors match in size.
    imfMRI = zeros(64*64*32,170,3);
    EEG = zeros(37,340000,3);
    for j=1:3
        I = niftiread([path,'/',list_subjects(i,:),'/func/sub-0',num2str(i),...
            '_task-auditoryoddballwithbuttonresponsetotargetstimuli_run-0',...
            num2str(j),'_bold.nii.gz']);
%         Uncomment for Visual Stimuli
%         I = niftiread([path,'\',list_subjects(i,:),'\func\sub-0',num2str(i),...
%             '_task-visualoddballwithbuttonresponsetotargetstimuli_run-0',...
%             num2str(j),'_bold.nii.gz']);
        
        imfMRI(:,:,j) = cast(reshape(I,[],170),'single');
        [oddball_fMRI{i,j}, standard_fMRI{i,j}] = trial_sep_eeg_fMRI(imfMRI(:,:,j), i, j,'path',path);
        
        load([path,'/',list_subjects(i,:),'/EEG/EEG_rereferenced_task01run0',num2str(j),'.mat'], 'data_reref');
        EEG(:,:,j) = cast(data_reref, 'single');
        [oddball_eeg{i,j}, standard_eeg{i,j}] = trial_sep_eeg_fMRI(reshape(EEG(:,:,j),[],170), i, j,'path', path);
        
    end
    
end


%!!!!!!!! CP rank is assumed to be 5. Needs some experimentation. Rank !!!!
% selection schemes should be explored.
R = 5;
w1 = 1;
w2 = 1;

EEG = {oddball_eeg,standard_eeg};
imfMRI = {oddball_fMRI, standard_fMRI};
shapes = {[37,2000,2,3,30],[64*64*32,2,3,30]};
% The featurs from sdf and acmtf utilize different toolboxes 
% (tensorlab and tensor_toolbox, respactively). Hence they are processed
% differently.
[features_sdf, features_acmtf, output] = apply_coupled_factorization_trials(EEG, imfMRI, shapes, R, w1, w2);



%%%%%%%%%%%% End of script
