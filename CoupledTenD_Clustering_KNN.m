% APPLY  Coupled tensor decomposition to multimodal sMRI and fMRI data and
% perform KNN clustering for different trails

clear; clc;
addpath("D:\Research\EEG-fMRI\EEG_fMRI_Code\Tensorlabs");

% For auditory stimulus
% fMRI_path = "D:\Research\EEG-fMRI\auditory_fMRI";
% label_path = "D:\Research\EEG-fMRI\auditory_label";  

% For visual stimulus
fMRI_path = "D:\Research\EEG-fMRI\visual_fMRI";
label_path = "D:\Research\EEG-fMRI\visual_label"; 

MRI_path = "D:\Research\EEG-fMRI\T1_MRI";

%%
label_list = ls(label_path);
MRI_list = ls(MRI_path);
fMRI_list = ls(fMRI_path);

cluster_performance = [];
[num_label, ~] = size(label_list);
for i = 3 : num_label
    sub_id = split(label_list(i, :), '_');
    MRI_name = join([sub_id{1}, "_T1w.nii.gz"], '');
    fMRI_name = join([sub_id{1}, "_", sub_id{2}, '_', sub_id{3}, '_bold.nii.gz'], '');
    imMRI = niftiread(fullfile(MRI_path, MRI_name));
    imfMRI = niftiread(fullfile(fMRI_path, fMRI_name));
    
    imMRI = imresize3(imMRI, [256,256,150]);
    imMRI = cast(imMRI, 'single');
    
    imfMRI = cast(imfMRI, 'single');
    activity = tdfread(fullfile(label_path, label_list(i, :)));
    Act_label = activity.Stimulus;
    [a, b] = KMeans_Clustering_Compare(imMRI, imfMRI, Act_label, 5);
    cluster_performance = cat(1, cluster_performance, [a, b]);
end

disp(mean(cluster_performance(:, 1)));
disp(std(cluster_performance(:, 2)));
disp(mean(cluster_performance(:, 2)));
disp(std(cluster_performance(:, 1)));

