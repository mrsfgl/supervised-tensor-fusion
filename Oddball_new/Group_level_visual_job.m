%-----------------------------------------------------------------------
% Job saved on 13-Apr-2021 17:29:26 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
clear matlabbatch
matlabbatch{1}.spm.stats.factorial_design.dir = {'D:\Research\Visual_Auditory_Processed\group_level_visual'};
data_dir = 'D:\\Research\\Visual_Auditory_Processed';

for s = 1 : nsub
    if s == 4
        continue
    end
    dname = fullfile(data_dir, sprintf('sub-%s\\visual_first_level', subs{s}));
    matlabbatch{1}.spm.stats.factorial_design.des.anovaw.fsubject(s).scans = {
        fullfile(dname, sprintf('con_0002.nii,1'))
        fullfile(dname, sprintf('con_0003.nii,1'))
        };
matlabbatch{1}.spm.stats.factorial_design.des.anovaw.fsubject(s).conds = [1 2];
end
cd('D:\Research\Visual_Auditory_Processed\group_level_visual');

matlabbatch{1}.spm.stats.factorial_design.des.anovaw.dept = 1;
matlabbatch{1}.spm.stats.factorial_design.des.anovaw.variance = 1;
matlabbatch{1}.spm.stats.factorial_design.des.anovaw.gmsca = 0;
matlabbatch{1}.spm.stats.factorial_design.des.anovaw.ancova = 0;
matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;
matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('Factorial design specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'Standard < Oddball';
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [-1 1];
matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'repl';
matlabbatch{3}.spm.stats.con.consess{2}.fcon.name = 'All Effect';
matlabbatch{3}.spm.stats.con.consess{2}.fcon.weights = [1 0 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625
                                                        0 1 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625];
matlabbatch{3}.spm.stats.con.consess{2}.fcon.sessrep = 'repl';
matlabbatch{3}.spm.stats.con.delete = 0;
spm_jobman('run', matlabbatch);