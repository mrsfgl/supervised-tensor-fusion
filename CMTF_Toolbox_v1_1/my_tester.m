
% ACMTF tests with soft coupling
s_coupl = 0;
sizes = [20,20,20,20];
ranks = [3];
dists = [0];
miss_rates = [0.1, 0.1];
noise_level = .5;
n_exps = 1;
n_ranks = length(ranks);
n_dists = length(dists);

modes = {[1,2,3], [1,4]};

corrs_acmtf = cell(n_dists, n_ranks, n_exps);
corrs_acmtf_wsc = cell(n_dists, n_ranks, n_exps);
corrs_cmtf = cell(n_dists, n_ranks, n_exps);
corrs_cp = cell(n_ranks, n_exps);
err_cp = cell(n_ranks, n_exps);
err_acmtf = cell(n_dists, n_ranks, n_exps);
err_acmtf_wsc = cell(n_dists, n_ranks, n_exps);
err_cmtf = cell(n_dists, n_ranks, n_exps);
for i_e = 1:n_exps
    for i_d = 1:n_dists
        for i_r = 1:n_ranks
            % Soft coupled data - hard coupled algorithm
            data = tester_cmtf_missing('size', sizes, 'R',ranks(i_r),'flag_soft',s_coupl,...
                'dist_coupled',dists(i_d),'rnd_seed', i_e, 'M', miss_rates,...
                'noise', noise_level);

            [err_cmtf{i_d,i_r,i_e}, corrs_cmtf{i_d,i_r,i_e}] = ...
                err_analysis_cmtf(data, modes, 1);

            if i_d == 1
                X = zeros(size(data.Xorig{1}));
                X(double(data.W{1})~=1) = data.Xorig{1}(find(data.W{1}~=1));
                X = tensor(X);
                P = data.W{1};
                K = cp_wopt(X,P,ranks(i_r));

                for n=1:3
                    Cmat = corr(K{n},data.Factrue{n});
                    corrs_cp{i_r,i_e}(n) = mean(max(abs(Cmat)));
                end
                K = full(K);
                trueval = data.Xorig{1}(find(data.W{1}==0));
                estval = K(find(data.W{1}==0));
                err_cp{i_r,i_e} = norm(estval - trueval)/length(estval);

                Y = zeros(size(data.Xorig{2}));
                Y(double(data.W{2})~=1) = data.Xorig{2}(find(data.W{2}~=1));
                [U,S,V] = svds(Y, ranks(i_r));
                corrs_pca{i_r,i_e}(1) = mean(max(abs(corr(U,data.Factrue{4}))));
                corrs_pca{i_r,i_e}(2) = mean(max(abs(corr(V,data.Factrue{5}))));
                trueval = data.Xorig{2}(find(data.W{2}==0));
                K = U*S*V';
                estval = K(double(data.W{2})==0);
                err_pca{i_r,i_e} = norm(estval - trueval)/length(estval);
            end
            % Soft coupled data - hard coupled algorithm
            data = tester_acmtf_missing('size', sizes, 'R',ranks(i_r),'flag_soft',s_coupl,...
                'dist_coupled',dists(i_d), 'rnd_seed', i_e, 'M', miss_rates,...
                'noise', noise_level);

            [err_acmtf{i_d,i_r,i_e}, corrs_acmtf{i_d,i_r,i_e}] = ...
                err_analysis_acmtf(data, modes, 1);

            % Soft coupled data - soft coupled algorithm
            data = tester_acmtf_wsc_missing('size', sizes, 'R',ranks(i_r),...
                'dist_coupled', dists(i_d),'rnd_seed',i_e, 'M', miss_rates,...
                'noise', noise_level, 'flag_soft', s_coupl);

            [err_acmtf_wsc{i_d,i_r,i_e}, corrs_acmtf_wsc{i_d,i_r,i_e}] = ...
                err_analysis_acmtf_wsc(data, modes, 1);
        end
    end
end

err = zeros(4,n_ranks,n_dists);
std_err = zeros(4,n_ranks,n_dists);
corrs = zeros(4,n_ranks,n_dists);
std_corr = zeros(4,n_ranks,n_dists);
for i_r = 1:n_ranks
    %% CP and PCA
    err(1,i_r,:) = mean(cellfun(@(x) x(1), err_cp(i_r,:)));
    std_err(1,i_r,:) = std(cellfun(@(x) x(1), err_cp(i_r,:)));
    corrs(1,i_r,:) = mean(cellfun(@mean,corrs_cp(i_r,:)));
    std_corr(1,i_r,:) = std(cellfun(@mean,corrs_cp(i_r,:)));
    for i_d = 1:n_dists
        %% CMTF
        err(2,i_r,i_d) = mean(cellfun(@(x) x(1), err_cmtf(i_d,i_r,:)));
        std_err(2,i_r,i_d) = std(cellfun(@(x) x(1), err_cmtf(i_r,:)));
        corrs(2,i_r,i_d) = mean(cellfun(@mean,corrs_cmtf(i_d,i_r,:)));
        std_corr(2,i_r,i_d) = std(cellfun(@mean,corrs_cmtf(i_d,i_r,:)));
        
        %% ACMTF
        err(3,i_r,i_d) = mean(cellfun(@(x) x(1), err_acmtf(i_d,i_r,:)));
        std_err(3,i_r,i_d) = std(cellfun(@(x) x(1), err_acmtf(i_r,:)));
        corrs(3,i_r,i_d) = mean(cellfun(@mean,corrs_acmtf(i_d,i_r,:)));
        std_corr(3,i_r,i_d) = std(cellfun(@mean,corrs_acmtf(i_d,i_r,:)));
        
        %% ACMTF-SC
        err(4,i_r,i_d) = mean(cellfun(@(x) x(1), err_acmtf_wsc(i_d,i_r,:)));
        std_err(4,i_r,i_d) = std(cellfun(@(x) x(1), err_acmtf_wsc(i_r,:)));
        corrs(4,i_r,i_d) = mean(cellfun(@mean,corrs_acmtf_wsc(i_d,i_r,:)));
        std_corr(4,i_r,i_d) = std(cellfun(@mean,corrs_acmtf_wsc(i_d,i_r,:)));
    end
end