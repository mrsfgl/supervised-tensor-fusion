function [oddball, standard] = trial_sep(imfMRI, path, i, j )
% 
%

task = tdfread([path,'\Sub00',num2str(i),'\func\sub-0',num2str(i),'_task-auditoryoddballwithbuttonresponsetotargetstimuli_run-0',num2str(j),'_events.tsv']);
trial_types = unique(task.trial_type,'rows');

id_oddball = find(prod(task.trial_type==trial_types(1,:),2));
id_oddball = unique(id_oddball);
n_oddball = length(id_oddball);
oddball_task_ids = zeros(n_oddball,2);

oddball = zeros(64,64,32,n_oddball);
last_oddball = 0;
for k=1:n_oddball
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
    oddball(:,:,:,k) = imfMRI(:,:,:,curr_oddball);
end
standard = zeros(64,64,32,n_oddball);
standard_task_ids = setdiff(1:170,[oddball_task_ids(:,1)-1;...
    oddball_task_ids(:,1);oddball_task_ids(:,1)+1]);
for k=1:n_oddball
    standard(:,:,:,k) = imfMRI(:,:,:,standard_task_ids(k));
end

% Reduce the trial number so that the experiments are faster.

standard = standard(:,:,:,1:min(5,n_oddball));
oddball = oddball(:,:,:,1:min(5,n_oddball));

end