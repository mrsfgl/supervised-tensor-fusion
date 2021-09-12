% Master script for data preprocessing in Visual Auditory Simultaneous EEG
% fMRI data analysis 

% Start from raw BIDS data downloaded from open neuron

rawpth = 'D:\Research\Visual_Auditory_EEG_fMRI';
sourpth = 'D:\Research\Visual_Auditory_Processed';

% mkdir(sourpth);

BIDS = spm_BIDS(rawpth);
subs = spm_BIDS(BIDS, 'subjects');
nsub = numel(subs);
subdir = cell(1, nsub);
for i = 1 : nsub
    subdir{i} = sprintf('sub-%s', subs{i});
end
% spm_mkdir(sourpth, subdir, {'anat', 'func'});


%%
for i = 1 : nsub
    if i == 4
        continue
    end
    f = spm_BIDS(BIDS, 'data', 'sub', subs{i}, 'modality', 'anat', 'type', 'T1w');
    spm_copy(f, fullfile(sourpth, subdir{i}, 'anat'), 'gunzip', true);
    
    f = spm_BIDS(BIDS, 'data', 'sub', subs{i}, 'modality', 'func', 'type', 'bold');
    spm_copy(f, fullfile(sourpth, subdir{i}, 'func'), 'gunzip', true);
end

%% Create Onset time (Visual)
onset_folder = fullfile(sourpth, 'Onset_time');
runs = spm_BIDS(BIDS, 'runs', 'modality', 'func', 'type', 'bold', 'task', 'visualoddballwithbuttonresponsetotargetstimuli');
nrun = numel(runs);
trialtypes = {'visual standard stimulus presentation', 'visual oddball stimulus presentation'};
for s = 1 : nsub
   if s == 4 
       continue
   end
   for r = 1 : nrun
      d = spm_load(char(spm_BIDS(BIDS, 'data', 'modality', 'func', 'type', 'events',...
          'task', 'visualoddballwithbuttonresponsetotargetstimuli', 'sub', subs{s}, 'run', runs{r})));
      clear conds
      for t = 1 : numel(trialtypes)
         conds.names{t} = trialtypes{t};
         conds.durations{t} = 0;
         conds.onsets{t} = d.onset(strcmpi(d.trial_type, trialtypes{t}));
      end
      save(fullfile(onset_folder, sprintf('sub-%s_visualtask_run-%s.mat', subs{s}, runs{r})), '-struct', 'conds');
   end
end


%% Create Onset time (Auditory)
onset_folder = fullfile(sourpth, 'Onset_time');
runs = spm_BIDS(BIDS, 'runs', 'modality', 'func', 'type', 'bold', 'task', 'auditoryoddballwithbuttonresponsetotargetstimuli');
nrun = numel(runs);
trialtypes = {'auditory standard stimulus presentation', 'auditory oddball stimulus presentation'};
for s = 1 : nsub
   if s == 4 
       continue
   end
   for r = 1 : nrun
      d = spm_load(char(spm_BIDS(BIDS, 'data', 'modality', 'func', 'type', 'events',...
          'task', 'auditoryoddballwithbuttonresponsetotargetstimuli', 'sub', subs{s}, 'run', runs{r})));
      clear conds
      for t = 1 : numel(trialtypes)
         conds.names{t} = trialtypes{t};
         conds.durations{t} = 0;
         conds.onsets{t} = d.onset(strcmpi(d.trial_type, trialtypes{t}));
      end
      save(fullfile(onset_folder, sprintf('sub-%s_auditorytask_run-%s.mat', subs{s}, runs{r})), '-struct', 'conds');
   end
end




%% Extract ROI XYZ from Level 2 analysis and extract data from level 1 SPM data

group_level_dir = 'D:\Research\Visual_Auditory_Processed\ROI';
load(fullfile(group_level_dir, 'Visual_ROI_Idx'));
ROI_XYZ = Region1;
for s = 1 : nsub
    if s == 4
        continue
    end
    subfolder = fullfile('D:\Research\Visual_Auditory_Processed\', sprintf('sub-%s', subs{s}), 'visual_ROI');
      subfolder2 = fullfile('D:\Research\Visual_Auditory_Processed\', sprintf('sub-%s', subs{s}), 'visual_first_level_13');
%     mkdir(subfolder);
%     load(fullfile('D:\Research\Visual_Auditory_Processed\', sprintf('sub-%s', subs{s}), 'visual_first_level\SPM.mat'));
%     vol = SPM.xY.P;
%     ROI = spm_get_data(vol, ROI_XYZ);
%     save(fullfile(subfolder, 'level_2_ROI.mat'), 'ROI');
    vol_std = spm_vol(fullfile(subfolder2, 'con_0002.nii'));
    vol_odd = spm_vol(fullfile(subfolder2, 'con_0003.nii'));
    ConROI_std = spm_get_data(vol_std, ROI_XYZ);
    ConROI_odd = spm_get_data(vol_odd, ROI_XYZ);
%     save(fullfile(subfolder, 'ConROI_std.mat'), 'ConROI_std');
%     save(fullfile(subfolder, 'ConROI_odd.mat'), 'ConROI_odd');
    save(fullfile(subfolder, 'Train_std.mat'), 'ConROI_std');
    save(fullfile(subfolder, 'Train_odd.mat'), 'ConROI_odd');
%     save(fullfile(subfolder, 'Test_std.mat'), 'ConROI_std');
%     save(fullfile(subfolder, 'Test_odd.mat'), 'ConROI_odd');
end


%% Resample EEG and separate trials
a = resample(data_reref', 200, 1000);
run1_event = load(fullfile(onset_folder, sprintf('sub-%02d_visualtask_run-%02d.mat',1, 1)));
std_ons = run1_event.onsets{1};
std_start = std_ons - 0.1;
a = a';
std_trial = [];
for i = 1 : length(std_start)
   start_idx = round(200 * std_start(i));
   end_idx = start_idx + 200 * 0.6;
   std_trial = cat(3, std_trial, a(1:34, start_idx : end_idx));
end


%%
figure
hold on 
tmp = std_trial(:, :, 10);
for i = 1 : 37
    plot(tmp(i, :));
end
hold off



%% Extract ROI XYZ from Level 2 analysis and extract data from level 1 SPM data (Auditory)

group_level_dir = 'D:\Research\Visual_Auditory_Processed\ROI';
load(fullfile(group_level_dir, 'Auditory_ROI_Idx.mat'));
ROI_XYZ = Region2;
for s = 1 : nsub
    if s == 4
        continue
    end
    subfolder = fullfile('D:\Research\Visual_Auditory_Processed\', sprintf('sub-%s', subs{s}), 'auditory_ROI');
      subfolder2 = fullfile('D:\Research\Visual_Auditory_Processed\', sprintf('sub-%s', subs{s}), 'auditory_first_level_1');
%     mkdir(subfolder);
%     load(fullfile('D:\Research\Visual_Auditory_Processed\', sprintf('sub-%s', subs{s}), 'visual_first_level\SPM.mat'));
%     vol = SPM.xY.P;
%     ROI = spm_get_data(vol, ROI_XYZ);
%     save(fullfile(subfolder, 'level_2_ROI.mat'), 'ROI');
    vol_std = spm_vol(fullfile(subfolder2, 'con_0002.nii'));
    vol_odd = spm_vol(fullfile(subfolder2, 'con_0003.nii'));
    ConROI_std = spm_get_data(vol_std, ROI_XYZ);
    ConROI_odd = spm_get_data(vol_odd, ROI_XYZ);
%     save(fullfile(subfolder, 'ConROI_std.mat'), 'ConROI_std');
%     save(fullfile(subfolder, 'ConROI_odd.mat'), 'ConROI_odd');
%     save(fullfile(subfolder, 'Train_std.mat'), 'ConROI_std');
%     save(fullfile(subfolder, 'Train_odd.mat'), 'ConROI_odd');
    save(fullfile(subfolder, 'Test_std.mat'), 'ConROI_std');
    save(fullfile(subfolder, 'Test_odd.mat'), 'ConROI_odd');
end