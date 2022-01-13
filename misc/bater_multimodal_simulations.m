
clear

rank = 3;
modes = {[1,2,3],[1,4]};
nsweep = 900;
nexps = 10;
n_levels = [0.005];
n_noise = length(n_levels);
sample_sizes = [500];
n_sample = length(sample_sizes);

%%%%%%%%%%%%%%% Generate Images %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M = length(modes);
sizes = [16,16,16,64]; % Dimensions of tensor samples.
for m=1:M
    d(m) = prod(sizes(modes{m}));
end

xtrain0 = cell(nexps,1);

% Initialization of
B_mean = cell(nexps); B_tens = cell(nexps, n_noise, n_sample);
B_mean_mc = cell(nexps);
B_vec = cell(nexps, n_noise, n_sample);
rmse_B = zeros(nexps, n_noise, n_sample, M); rmse_val = zeros(nexps, n_noise, n_sample);
norm_err_B = zeros(nexps, n_noise, n_sample, M); norm_1_B = zeros(nexps, n_noise, n_sample, M);
rmse_B_mc = zeros(nexps, n_noise, n_sample, M); rmse_val_mc = zeros(nexps, n_noise, n_sample);
norm_err_B_mc = zeros(nexps, n_noise, n_sample, M); norm_1_B_mc = zeros(nexps, n_noise, n_sample, M);
rmse_B_tens = zeros(nexps, n_noise, n_sample, M);
norm_err_B_tens = zeros(nexps, n_noise, n_sample, M);
norm_1_B_tens = zeros(nexps, n_noise, n_sample, M);
rmse_B_vec = zeros(nexps, n_noise, n_sample); rmse_val_vec = zeros(nexps, n_noise, n_sample);
norm_err_vec = zeros(nexps, n_noise, n_sample); norm_1_vec = zeros(nexps, n_noise, n_sample);

parfor i_exps = 1:nexps
    xtrain0{i_exps} = cell(M,1);
    B_0 = zeros(sum(d),1);
    im = generate_image([sizes(1:3),16,64], 4, {[1,2,3],[4,5]}, rank);
    for m=1:M
        %% Data for image m
        idx = (sum(d(1:m-1))+1):sum(d(1:m)); % Select the indices corresponding to the current modality.
        B_0(idx) = im{m}(:);
    end
    for i_sample = 1:n_sample
        disp(['Sample size is ',num2str(sample_sizes(i_sample))])
        n = sample_sizes(i_sample); % Number of samples.
        ytrain = zeros(n,1);
        X = generate_multimodal_data(sizes, rank, n, modes);
        for m=1:M
            xtrain0{i_exps}{m} = reshape(X{m}.orig,n,[]); % Create data
            xtrain0{i_exps}{m} = xtrain0{i_exps}{m}/diag(std(xtrain0{i_exps}{m}));
            %% Generate labels %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            ytrain = ytrain + xtrain0{i_exps}{m}*im{m}(:);
        end
        for i_noise = 1:n_noise
            %%%%%%%%%%%%%%%%%%%%%%%% SIMULATION_TENSOR %%%%%%%%%%%%%%%%%%%%%%%%
            noise_std = n_levels(i_noise); % Standard deviation of noise
            
%             xtrain = cell(M,1);
%             for m=1:M
%                 N = diag(noise_std.*sqrt(sqrt(sum(xtrain0{i_exps}{m}.^2,2))))*randn(n, prod(sizes(modes{m}))); % Create noise
%                 xtrain{m} = reshape(xtrain0{i_exps}{m}+N, [n,sizes(modes{m})]);
%             end
            ytrain = ytrain + noise_std*randn(n,1);
            ytrain = ytrain/std(ytrain);
            z_train = zeros(n, 1);
            
            %% Apply the vector based method to the concatenation of both data.
            xt = [];
            for m = 1:M
                xt = [xt, t2m(xtrain0{i_exps}{m},1)];
            end
            [B, FitInfo] = lassoglm([xt,z_train], ytrain, 'normal', 'CV', 10, 'Standardize', true);
            
            ypred =  xt*B(1:end-1,FitInfo.IndexMinDeviance);
            rmse_val(i_exps,i_noise,i_sample) = sqrt(mean((ypred-ytrain).^2));
            
            B_vec{i_exps,i_noise,i_sample} = B(1:end-1,FitInfo.IndexMinDeviance)*std(ytrain);
%             min_B = min(B_vec{i_exps,i_noise,i_sample}, [], 'all');
%             max_B = max(B_vec{i_exps,i_noise,i_sample}, [], 'all');
%             B_vec{i_exps,i_noise,i_sample} = (B_vec{i_exps,i_noise,i_sample}-min_B)./(max_B-min_B);
            rmse_B_vec(i_exps,i_noise,i_sample) = sqrt(norm(B_vec{i_exps,i_noise,i_sample}-B_0)^2/prod(sizes(modes{m})));
            norm_err_vec(i_exps,i_noise,i_sample) = sum((B_vec{i_exps,i_noise,i_sample}-B_0).^2,'all')/norm(B_0).^2;
            norm_1_vec(i_exps,i_noise,i_sample) = sum(abs(B_vec{i_exps,i_noise,i_sample}-B_0),'all')/norm(B_0,1);
            
            %% Apply the tensor based method for each modality separately
            for m = 1:M
                res = bater(z_train, xtrain0{i_exps}{m}, ytrain, nsweep, rank);
                
                % Evaluate
                B_tens{i_exps,i_noise,i_sample} = zeros(sizes(modes{m}));
                for i = 1:nsweep
                    beta = res(i).beta_store;
                    B = zeros(sizes(modes{m}));
                    for j = 1:rank
                        B = B + reshape(getouter(beta(j,:)),sizes(modes{m}));
                    end
                    B_tens{i_exps,i_noise,i_sample} = B_tens{i_exps,i_noise,i_sample}+B./nsweep;
                end
                
                out = eval_reg(B_tens{i_exps,i_noise,i_sample}, im{m}*std(ytrain));
                rmse_B_tens(i_exps,i_noise,i_sample,m) = out.rmse_B;
                norm_err_B_tens(i_exps,i_noise,i_sample,m) = out.norm_err_B;
                norm_1_B_tens(i_exps,i_noise,i_sample,m) = out.norm_1_B;
                disp(['RMSE of B using tensor based method at noise index ',...
                    num2str(i_noise),', sample index ',num2str(i_sample),...
                    ', and modality ',num2str(m),': ',num2str(out.rmse_B)])
            end
            
            %% Apply the multimodal method with concatenation
            res = bater_multimodal_concat(z_train, xtrain0{i_exps}, ytrain, nsweep, rank);
            %%%%%%%%%%%%%%%%%%%%%%% Evaluate %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            ypred = zeros(nsweep,n);
            rms = zeros(M,nsweep);
            for m=1:M
                B_mean{i_exps}{m}{i_noise,i_sample} = zeros(sizes(modes{m}));
            end
            for i = 1:nsweep
                beta = res(i).beta_store;
                for m = 1:M
                    B = zeros(sizes(modes{m}));
                    for j=1:rank
                        B = B + reshape(getouter(beta{m}(j,:)),sizes(modes{m}));
                    end
                    B_mean{i_exps}{m}{i_noise,i_sample} = B_mean{i_exps}{m}{i_noise,i_sample}+B./nsweep;
                end
            end
            
            ypredall = zeros(n,1);
            for m = 1:M
                out = eval_reg(B_mean{i_exps}{m}{i_noise,i_sample}, im{m});
                rmse_B(i_exps,i_noise,i_sample,m) = out.rmse_B;
                norm_err_B(i_exps,i_noise,i_sample,m) = out.norm_err_B;
                norm_1_B(i_exps,i_noise,i_sample,m) = out.norm_1_B;
                disp(['RMSE of B at noise index ',num2str(i_noise),', sample',...
                    ' index ',num2str(i_sample),', and modality ',num2str(m),...
                    ': ',num2str(out.rmse_B)])
                
                ypredall = ypredall + t2m(xtrain0{i_exps}{m},1)*B_mean{i_exps}{m}{i_noise,i_sample}(:);
            end
            
            rmse_val(i_exps,i_noise,i_sample) = sqrt(mean((ypredall-normalize(ytrain)).^2));
            
            %% Apply the multimodal method
            res = bater_multimodal(z_train, xtrain0{i_exps}, ytrain, nsweep, rank, modes);
            
            %%%%%%%%%%%%%%%%%%%%%%% Evaluate %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            ypred_mc = zeros(nsweep,n);
            rms_mc = zeros(M,nsweep);
            for m=1:M
                B_mean_mc{i_exps}{m}{i_noise,i_sample} = zeros(sizes(modes{m}));
            end
            for i = 1:nsweep
                beta = res(i).beta_store;
                for m = 1:M
                    B = zeros(sizes(modes{m}));
                    for r=1:rank
                        B = B + reshape(getouter(beta(r,modes{m})),sizes(modes{m}));
                    end
                    B_mean_mc{i_exps}{m}{i_noise,i_sample} = B_mean_mc{i_exps}{m}{i_noise,i_sample}+B./nsweep;
                end
            end
            
            ypredall = zeros(n,1);
            for m = 1:M
                out = eval_reg(B_mean_mc{i_exps}{m}{i_noise,i_sample}, im{m});
                rmse_B_mc(i_exps,i_noise,i_sample,m) = out.rmse_B;
                norm_err_B_mc(i_exps,i_noise,i_sample,m) = out.norm_err_B;
                norm_1_B_mc(i_exps,i_noise,i_sample,m) = out.norm_1_B;
                disp(['RMSE of B at noise index ',num2str(i_noise),', sample',...
                    ' index ',num2str(i_sample),', and modality ',num2str(m),...
                    ': ',num2str(out.rmse_B)])
                
                ypredall = ypredall + t2m(xtrain0{i_exps}{m},1)*B_mean_mc{i_exps}{m}{i_noise,i_sample}(:);
            end
            
            rmse_val_mc(i_exps,i_noise,i_sample) = sqrt(mean((ypredall-normalize(ytrain)).^2));
        end
    end
end
%
% write.csv(rms,file="n2k_yrmse_train.csv")
% write.csv(ypred,file="n2k_ypred.csv")
% write.csv(B.mean,file = "n2k_Bmean.csv")
% write.csv(rmse(B.mean,B_0),file="n2k_Brmse.csv")
% %write.csv(B,file = "B_samples.csv")
% %write.csv(res,file = "Result.csv")
% write.csv(res$alpha.store,file="n2k_eta.csv")
%
% %jpeg(file="Recovered_plot.jpg")
% image(as.matrix(B.mean), axes = FALSE, useRaster = T, col = grey(seq(0, 1, length = 256)))
% %dev.off()