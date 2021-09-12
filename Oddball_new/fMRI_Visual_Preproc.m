% List of open inputs
nrun = X; % enter the number of runs here
jobfile = {'D:\Research\EEG-fMRI\EEG_fMRI_Code\Full_DataProc\fMRI_Visual_Preproc_job.m'};
jobs = repmat(jobfile, 1, nrun);
inputs = cell(0, nrun);
for crun = 1:nrun
end
spm('defaults', 'FMRI');
spm_jobman('run', jobs, inputs{:});
