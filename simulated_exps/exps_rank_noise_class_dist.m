
sizes = [30, 20, 10, 50];
modes = {[1,2,3],[1,4]};
n_exps = 10;
n_samples = 20;
folds = 5;
class_var = [0.2];
% M = [.1, .1];
M = {[.2,.2]};

noise_levels = [3]; % dB
% noise_levels = [.1:.2:1];
ranks = [3];
% ranks = [3:2:9];
class_dists = {[.5,.5,.5,2],[1,.5,.5,2],[1,1,1,2],[.5,.5,.5,3]};
% class_dists = [.1,.5,1,2,3];
n_noise = length(M);
n_ranks = length(ranks);
n_class = length(class_dists);

results = cell(n_exps, n_ranks, n_class, n_noise);
for i_exps = 1:n_exps
    for i=1:n_ranks
        lambdas = {ones(1,ranks(i)), ones(1,ranks(i))};
        for j=1:n_noise
            for k = 1:n_class
                [X, A] = create_coupled_supervised('lambdas', lambdas, ...
                    'noise', noise_levels, 'rnd_seed', 123*i_exps, ...
                    'class_distance', class_dists{k}, 'size', sizes,...
                    'modes', modes, 'n_samples', n_samples, 'class_var',...
                    class_var, 'M', M{j});
                
                results{i_exps, i, k, j} = cpld_class(X, A, folds);
            end
        end
    end
end