
clear; clc
addpath 'D:\Research\EEG-fMRI\EEG_fMRI_Code\CMTF_Toolbox_v1_1'
addpath 'D:\Research\EEG-fMRI\EEG_fMRI_Code\CoupledDecomposition'
sizes = [30, 20, 10, 50];
modes = {[1,2,3],[4, 3]};
n_exps = 1;
n_samples = 50;
class_var = [1];


noise_levels = [0]; % dB
M = [.1,.1];
% noise_levels = [.1:.2:1];
ranks = 3;
class_dists = [0.5, 0, 0, 0.5];
% [1,.5,.5,2],[1,1,1,2],[.5,.5,.5,3]};


lambdas = {ones(1,ranks), ones(1,ranks)};
[X, A] = create_coupled_supervised_modified('lambdas', lambdas, ...
                    'noise', noise_levels, 'rnd_seed', 123, ...
                    'class_distance', class_dists, 'size', sizes,...
                    'modes', modes, 'n_samples', n_samples, 'class_var',...
                    class_var);

