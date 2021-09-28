
clear 
clc
sizes = [30, 20, 10, 20];
modes = {[1,2,3],[4,3]};
n_exps = 10;
n_samples = 50;
folds = 5;
class_var = [1];
% M = [.1, .1];
M = {[0,0]};

noise_levels = [1,5,9,13]; % dB
% noise_levels = [.1:.2:1];
ranks = [3];
% ranks = [3:2:9];
class_dists = {[.3,.3,.3,.3],[0,0,1,0],[0,0,0,1],[0,0,.5,.5],[1,0,0,0]};%, 0.4.*ones(1,length(sizes)), ...
    %0.6.*ones(1,length(sizes)), 0.8.*ones(1,length(sizes))};
% class_dists = [.1,.5,1,2,3];
n_miss = length(M);
n_noise = length(noise_levels);
n_ranks = length(ranks);
n_class = length(class_dists);

results = cell(n_exps, n_ranks, n_class, n_noise, n_miss);
parfor i_exps = 1:n_exps
    for i=1:n_ranks
%         lambdas = {[0,1,1], [1,0,1]};
        lambdas = {[1,1,1], [1,1,1]};
        for j=1:n_noise
            for k = 1:n_class
                for m=1:n_miss
                    [X, A] = create_coupled_supervised('lambdas', lambdas, ...
                        'noise', noise_levels(j), 'rnd_seed', 123*i_exps, ...
                        'class_distance', class_dists{k}, 'size', sizes,...
                        'modes', modes, 'n_samples', n_samples, 'class_var',...
                        class_var, 'M', M{m});

                    results{i_exps, i, k, j, m} = cpld_class(X, A, folds);
                end
            end
        end
    end
end

% if length(M)>1 && length(noise_levels)==1
%     results = permute(results, [1,2,5,4,3]);
%     out = plot_noise_exps(results(:,:,:,1), cellfun(@(x) x(1),M), 'miss_levels');
% else
%     out = plot_noise_exps(results, noise_levels,'noise_levels');
% end
save('exps_noise_class_discr.mat', 'results', 'M', 'noise_levels', 'ranks', 'class_dists')