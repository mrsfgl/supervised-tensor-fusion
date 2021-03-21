% Converting fMRI images to 3D using fMRI_Split function
% Dependency: fMRI_Split.m; SPM 12;

% Starting from Subject 02, and assume there is no folder in the system
% before

clear; clc;

main_dir = 'D:\Research\Multimodal_EEG_fMRI_Proc\Processed_Data\fMRI';
source_dir = 'D:\Research\Multimodal_EEG_fMRI_Face';

for i = 1 : 16
    tmpfile_folder = join([source_dir, filesep, 'sub-', num2str(i, '%02d'), filesep, 'ses-mri', filesep,...
        'func'], '');
    disp(tmpfile_folder);
    for j = 1 : 9
       filename = join(['sub-', num2str(i, '%02d'), '_ses-mri_task-facerecognition_run-', num2str(j, '%02d'), '_bold.nii.gz'], '');
       filename2 = join(['sub-', num2str(i, '%02d'), '_ses-mri_task-facerecognition_run-', num2str(j, '%02d'), '_bold.nii'], '');
       disp(filename);
       gunzip(join([tmpfile_folder, filesep, filename], ''));
       imgfile = join([tmpfile_folder, filesep, filename2], '');
       outdir = join([main_dir, filesep, 'Sub', num2str(i, '%02d'), filesep, 'Run', num2str(j, '%02d')], '');
       mkdir(outdir);
       fMRI_Split(imgfile, outdir);
    end
end