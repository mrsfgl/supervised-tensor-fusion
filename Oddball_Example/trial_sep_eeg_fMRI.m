function [oddball, standard] = trial_sep_eeg_fMRI(I, tsv_name, num_trials, varargin )
%TRIAL_SEP_EEG_FMRI separates EEG or fMRI slices 
% corresponding to oddball and standard stimulus.
%
%   [oddball, standard] = TRIAL_SEP_EEG_FMRI(I, i, j) separates data I 
%   extracted from subject i and run j.
%   
%   [oddball, standard] = TRIAL_SEP_EEG_FMRI(I, i, j,'param',value,...) specifies additional
%   parameters for the method. Specifically..., 
%   
%   'num_trials',   specifies the number of trials to be extractedfrom each run for each 
%   stimulus type. {5}
%   
%   'stim_type',    specifies the stimulus type. {'auditory'}
%       'visual'  Visual oddball stimulus.
%   
%   'path',    the directory of EEG-fMRI data. 
%
% Input:   I: a matrix where columns correspond to 170 time slices for fMRI
%          and epochs for EEG.
%           
% Output:  f: function value of the combined objective function.
%          g: a vector corresponding to the gradient.
% 
% Seyyid Emre Sofuoglu
% sofuogluatmsu.edu

% Parse inputs
param = inputParser;
% param.addParameter('num_trials', 5, @(x) isinteger(x) && x>=0);
% param.addParameter('stim_type', 'visual');
% param.addParameter('path', '/egr/research/sigimprg/Emre/databases/EEG-fMRI/EEG_FMRI');
param.parse(varargin{:});
%
% path = param.Results.path;
% num_trials = param.Results.num_trials;
% stim_type = param.Results.stim_type;
%

sz = size(I);
% task = tdfread([path,'/Sub00',num2str(i),'/func/sub-0',num2str(i),'_task-',stim_type,'oddballwithbuttonresponsetotargetstimuli_run-0',num2str(j),'_events.tsv']);
task = tdfread(tsv_name);
trial_types = unique(task.trial_type,'rows');

id_oddball = find(prod(task.trial_type==trial_types(1,:),2));
id_oddball = unique(id_oddball);
n_oddball = length(id_oddball);
oddball_task_ids = zeros(n_oddball,2);

oddball = ones(sz(1),n_oddball);
last_oddball = 0;
for k=1:n_oddball
    if id_oddball(k)+3>size(task.onset,1)
        break
    end
    onsets = [str2double(task.onset(id_oddball(k),:)),...
        str2double(task.onset(id_oddball(k)+1,:)),...
        str2double(task.onset(id_oddball(k)+2,:)),...
        str2double(task.onset(id_oddball(k)+3,:))];
    if contains(task.trial_type(id_oddball(k)+1,:), 'behavioral')
        oddball_task_ids(k,:) = (floor([onsets(1), onsets(3)-1]));
    elseif contains(task.trial_type(id_oddball(k)+1,:), 'nan', 'IgnoreCase', true)
        if contains(task.trial_type(id_oddball(k)+2,:), 'behavioral')
            oddball_task_ids(k,:) = (floor([onsets(1), onsets(4)-1]));
        end
    else
        oddball_task_ids(k,:) = (floor([onsets(1), onsets(2)-1]));
    end
    curr_oddball = floor(oddball_task_ids(k,1)/2)+mod(oddball_task_ids(k,1),2);
    if curr_oddball == last_oddball
        curr_oddball = curr_oddball+1;
    end
    last_oddball = curr_oddball;
    oddball_task_ids(k,2) = round((oddball_task_ids(k,2)-oddball_task_ids(k,1))/2)+curr_oddball;
    oddball_task_ids(k,1) = curr_oddball;
    oddball(:,k) = I(:,curr_oddball);
end
standard = zeros(sz(1),n_oddball);
standard_task_ids = setdiff(1:sz(2),[oddball_task_ids(:,1)-1;...
    oddball_task_ids(:,1);oddball_task_ids(:,1)+1]);
for k=1:n_oddball
    standard(:,k) = I(:,standard_task_ids(k));
end

% Reduce the trial number so that the experiments are faster.

standard = standard(:,1:min(num_trials, n_oddball));
oddball = oddball(:,1:min(num_trials, n_oddball));

end