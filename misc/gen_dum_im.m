

%%%%%%%%%%%%%%% Generate Image 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = zeros(64,64);
m(30:34,22:42) = 1;
m(22:42,30:34) = 1;
m(6:10,6:10) = 1;
m(6:10,54:58) = 1;
m(54:58,6:10) = 1;
m(54:58,54:58) = 1;

% m = zeros(64,64,64);
% m(30:34,22:42,30:34) = 1;
% m(22:42,30:34,30:34) = 1;
% m(30:34,30:34,22:42) = 1;
% m(6:10,6:10,6:10) = 1;
% m(6:10,54:58,6:10) = 1;
% m(6:10,6:10,54:58) = 1;
% m(6:10,54:58,54:58) = 1;
% m(54:58,6:10,6:10) = 1;
% m(54:58,54:58,6:10) = 1;
% m(54:58,6:10,54:58) = 1;
% m(54:58,54:58,54:58) = 1;
sizes = [64,64]; % Dimensions of matrix samples.

n_levels = [0.001, 0.005, 0.01, 0.05, 0.1]; n_noise = length(n_levels);
sample_sizes = [50, 100, 300, 500, 700, 1000]; n_sample = length(sample_sizes);
B_mean = cell(n_noise, n_sample); B_vec = cell(n_noise, n_sample);
rmse_B = zeros(n_noise, n_sample); rmse_val = zeros(n_noise, n_sample);
norm_err_B = zeros(n_noise, n_sample); norm_1_B = zeros(n_noise, n_sample);
rmse_B_vec = zeros(n_noise, n_sample); rmse_val_vec = zeros(n_noise, n_sample);
norm_err_vec = zeros(n_noise, n_sample); norm_1_vec = zeros(n_noise, n_sample);
for i_sample = 1:n_sample
    disp(['Sample size is ',num2str(sample_sizes(i_sample))])
    for i_noise = 1:n_noise
        %%%%%%%%%%%%%%%%%%%%%%%% SIMULATING IMAGES %%%%%%%%%%%%%%%%%%%%%%%%%%%
        n = sample_sizes(i_sample); % Number of samples.
        noise_std = n_levels(i_noise); % Standard deviation of noise
        
        xtrain0 = randn(n,prod(sizes)); % Create data
        N = diag(noise_std.*sqrt(sqrt(sum(xtrain0.^2,2))))*randn(n, prod(sizes)); % Create noise
        xtrain = reshape(xtrain0+N, [n,sizes]);
        
        % xtrain0 = csvread('xtrain.csv');
        % n = size(xtrain0,1)-1;
        % xtrain0 = reshape(xtrain0(2:end,2:end),n,4096);
        % xtrain = zeros(n,64,64);
        % for i = 1:n
        %     xtrain(i,:,:) = reshape(xtrain0(i,:),64,64)+0.1*randn(64,64)*sqrt(norm(xtrain0(i,:)));
        % end
        % ytrain = csvread("ytrain.csv");
        % ytrain = ytrain(2:end,2:end);
        
%         n = 1000; % Number of samples.
%         noise_std = 0; % Standard deviation of noise
% 
%         xtrain0 = randn(n,prod(sizes)); % Create 
%         xtrain = zeros([n,sizes]);
%         for i = 1:n
%             xtrain(i,:,:,:) = reshape(xtrain0(i,:),sizes)+(noise_std*sqrt(norm(xtrain0(i,:)))).*randn(sizes);
%         end

        %%%%%%%% Generate data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %% Data for image 1
        B_0 = m(:);
        ytrain = xtrain0*B_0;
        z_train = zeros(n, 1);

        %%%%% Apply the vector based method
        [B, FitInfo] = lassoglm([t2m(xtrain,1),z_train], ytrain, 'normal', 'CV', 10, 'Standardize', true);
        
        B_vec{i_noise,i_sample} = B(1:end-1,FitInfo.IndexMinDeviance);
        min_B = min(B_vec{i_noise,i_sample}, [], 'all');
        max_B = max(B_vec{i_noise,i_sample}, [], 'all');
        B_vec{i_noise,i_sample} = (B_vec{i_noise,i_sample}-min_B)./(max_B-min_B);
        rmse_B_vec(i_noise, i_sample) = sqrt(norm(B_vec{i_noise,i_sample}-B_0)^2/prod(sizes));
        norm_err_vec(i_noise, i_sample) = sum((B_vec{i_noise,i_sample}-B_0).^2,'all')/norm(m(:)).^2;
        norm_1_vec(i_noise, i_sample) = sum(abs(B_vec{i_noise,i_sample}-B_0),'all')/norm(m(:),1);
        
        ypred =  t2m(xtrain,1)*B_vec{i_noise,i_sample};
        
        rmse_val_vec(i_noise, i_sample) = sqrt(mean((ypred-normalize(ytrain)).^2));
        
        %%%%% Apply the tensor based method        
        res = bater(z_train, xtrain, ytrain, 900, 10);

        %%%%%%%%%%%%%%%%%%%%%%%Calculate RMSE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        rank = 10;
        %burn  =  1
        %nskip  = 5

        nsweep = length(res);
        ypred = zeros(nsweep,n);
        rms = zeros(1,nsweep);
        B_mean{i_noise,i_sample} = zeros(res(1).p);
        for i = 1:nsweep
            beta = res(i).beta_store;
            B = zeros(res(1).p);
            for j=1:rank
                B = B + reshape(getouter(beta(j,:)),res(1).p);
            end
            B_mean{i_noise,i_sample} = B_mean{i_noise,i_sample}+B./nsweep;
            
            %       ypred(i,:) =  t2m(res(i).Xt,1)*B(:);
            rms(i) = sqrt(mean((ypred(i,:)'-res(i).obs).^2));
        end
        
        min_B = min(B_mean{i_noise,i_sample}, [], 'all');
        max_B = max(B_mean{i_noise,i_sample}, [], 'all');
        B_mean{i_noise,i_sample} = (B_mean{i_noise,i_sample}-min_B)./(max_B-min_B);
        rmse_B(i_noise, i_sample) = sqrt(sum((B_mean{i_noise,i_sample}-m).^2,'all')/prod(sizes));
        norm_err_B(i_noise, i_sample) = sum((B_mean{i_noise,i_sample}-m).^2,'all')/norm(m(:)).^2;
        norm_1_B(i_noise, i_sample) = sum(abs(B_mean{i_noise,i_sample}-m),'all')/norm(m(:),1);
        disp(['RMSE of B at noise index ',num2str(i_noise),': ',num2str(rmse_B(i_noise, i_sample))])
        
        ypredall =  t2m(res(i).Xt,1)*B_mean{i_noise,i_sample}(:);
        
        rmse_val(i_noise, i_sample) = sqrt(mean((ypredall-res(1).obs).^2));
        disp(['RMSE at iteration ',num2str(i_noise),': ',num2str(rmse_val(i_noise, i_sample))])
    end
end