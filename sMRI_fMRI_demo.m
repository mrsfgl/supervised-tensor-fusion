
clear
clc
addpath(genpath('M:\Documents\MATLAB\tboxunzipped\tensorlab'))

% Specifying data path should be enough to read and process all data. Check
% if all subjects have T1 images and 3 auditory oddball fMRI.
path = '\\cifs.egr.msu.edu\research\sigimprg\Emre\databases\EEG-fMRI\EEG_FMRI';
list_subjects = ls(path);
list_subjects = list_subjects(3:end,:);
n_subjects = size(list_subjects,1);

%!!!!!!!! CP rank is assumed to be 5. Needs some experimentation. Rank !!!!
% selection schemes should be explored.
R = 5;

oddball = cell(n_subjects, 3);
standard = cell(n_subjects, 3);
F = cell(1,n_subjects);
output = F;
for i = 1:n_subjects
    % For each subject extract sMRI and 3 auditory task data. Apply
    % necessary reshaping operations such that factors match in size.
    I = imresize3(niftiread([path,'\',list_subjects(i,:),'\anat\sub-0',...
        num2str(i),'_anat_sub-0',num2str(i),'_T1w.nii.gz']),...
        [256,256,150]);
    imsMRI = cast(I,'single');
    imfMRI = zeros(64,64,32,170,3);
    n_tot = 0;
    for j=1:3
        I = niftiread([path,'\',list_subjects(i,:),'\func\sub-0',num2str(i),...
            '_task-auditoryoddballwithbuttonresponsetotargetstimuli_run-0',...
            num2str(j),'_bold.nii.gz']);
        imfMRI(:,:,:,:,j) = cast(I,'single');
        [oddball{i,j}, standard{i,j}] = trial_sep(imfMRI(:,:,:,:,j), path, i, j);
%         for k = 1:size(oddball{i,j},4)
%             % Learn factors for each trial. These factors are trial-specific.
%             % Hence they can be utilized in a CP-STM framework.
%             [oddball_features{i}{n_tot+k}, output{i}{n_tot+k}] = extractFactors(imsMRI, oddball{i,j}(:,:,:,k),R);
%             % Two classes of trials, oddball and standard. The cell arrays
%             % can be reshaped into a tensor. They are only in cell format
%             % to separate subject-specific data which currently have no
%             % use.
%             [standard_features{i}{n_tot+k}, output{i}{n_tot+k}] = extractFactors(imsMRI, standard{i,j}(:,:,:,k), R);
%         end
        n_tot = n_tot+size(oddball{i,j},4);
    end
    
end

function [F, output] = extractFactors(imsMRI, imfMRI, R)
% [F, output] = extractFactors(imsMRI, imfMRI)
% Extracts coupled and uncoupled factors of structural and functional MRI.
sz_s = size(imsMRI);
sz_f = size(imfMRI);

% Define model variables
model = struct;
model.variables.u1 = randn(sz_s(1),R);
model.variables.u2 = randn(sz_s(2),R);
model.variables.u3 = randn(sz_s(3),R);

% Define subsampling matrices
r_horizontal = ceil(sz_s(1)/sz_f(1));
avM = kron(eye(sz_f(1)), ones(r_horizontal,1))/r_horizontal;
r_vertical = ceil(sz_s(3)/sz_f(3));
avH = kron(eye(sz_f(3)), ones(r_vertical,1))/r_vertical;
avH = avH(1:sz_s(3),:);

% Define factor relations to variables.
model.factors.U1 = {'u1'};
model.factors.U2 = {'u2'};
model.factors.U3 = 'u3';
model.factors.V1 = {'u1', @(z, task) struct_matvec(z,task,avM')};
model.factors.V2 = {'u2', @(z, task) struct_matvec(z,task,avM')};
model.factors.V3 = {'u3', @(z, task) struct_matvec(z,task,avH')};

% Define tensors and their factors to be learned.
model.factorizations.tensor1.data = imsMRI;
model.factorizations.tensor1.cpd = {'U1','U2','U3'};

model.factorizations.tensor2.data = imfMRI;
model.factorizations.tensor2.cpd = {'V1','V2','V3'};

% Optimize
options.Display = 50;
options.TolX = 10^-6;
options.TolFun = 10^-6;
options.MaxIter = 100;
[F, output] =  sdf_minf(model, options);

end