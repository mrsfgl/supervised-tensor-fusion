function out = bater_multimodal(z_train, x_train, y_train, nsweep, rank, modes) 
%%
%

n = length(y_train);
M = length(x_train);
for m=1:M
    if ndims(x_train{m})-1 ~= length(modes{m})
        error("Mismatch between data and variable: `modes`.")
    end
    sz(modes{m}) = size(x_train{m}, 2:ndims(x_train{m}));
end
d = length(sz); 
len_alpha = 10;
pgamma = size(z_train, 2); %ztrain\in R^{n x pgamma}

%% Standardize
my = mean(y_train,1);
sy = std(y_train,[],1);
obs = (y_train-my)/sy;

Xt = cell(1,M);
for m=1:M
    sx = std(x_train{m});
    sx(sx==0) = 1;
    Xt{m} = (x_train{m}-repmat(mean(x_train{m},1),[n,...
        ones(1,length(modes{m}))]))/diag(sx);
end
%% MCMC Setup
ZZ = z_train'*z_train;

%% Initialize 
% Hyperparameters
a_lam = repmat(3, 1, rank);
b_lam = a_lam.^(1/(2*d));
phi_alpha = repmat(1/rank, 1, rank);

b_vphi = phi_alpha(1)*rank^(1/4);

c0 = 0;
s0 = 1;
a_t = 2.5/2;
b_t = 2.5/2 * s0^2;
tau2 = 1 / gamrnd(a_t, 1./b_t, 1); % initialize tau2

lambda = gamrnd(a_lam(1), 1./b_lam(1), rank, d); % initialize lambda with size rank x d
omega = cell(d,1);
for j = 1:d
    omega{j} = exprnd(0.5*(a_lam(1)/b_lam(1)), rank, sz(j));% initialize omega
end
beta = cell(rank,d);
for j = 1:d
    for r = 1:rank
        beta{r,j} = randn(sz(j),1); % initialize beta
    end
end
alpha_grid = linspace(rank^(-d), rank^(-0.1), len_alpha);  % grid of alpha values

%% MCMC run 
tt = tic;
alpha_store = cell(nsweep,1);
c0_store = cell(nsweep,1);
gam_store = cell(nsweep,1);
tau2_store = cell(nsweep,1);
phi_store = cell(nsweep,1);
varphi_store = cell(nsweep,1);
beta_store = cell(nsweep,1);
omega_store = cell(nsweep,1);
lambda_store = cell(nsweep,1);
hyppar_store = cell(nsweep,1);
score_store = cell(nsweep,1);
tens_mean = cell(M,1);
for sweep = 1:nsweep
    pred_rest = zeros(n,1);
    for m = 1:M
        tens_mean{m} = getmean(Xt{m}, beta(:,modes{m}), rank, sz(modes{m}), []);
        pred_rest = pred_rest+tens_mean{m};
    end
        
    %% update gamma (scalar predictor coefficients)
    Sig_g = eye(pgamma) + ZZ / tau2; % eye(pgamma)-z_train*z_train'/(tau2+z_train'*z_train);
    mu_g = Sig_g \ (z_train'*(obs-c0-pred_rest)/tau2);
    gam = mu_g + chol(Sig_g) * randn(pgamma,1);
        
    %% update alpha (intercept) 
    pred_mean = z_train * gam;
    mu_c0 = mean(obs-pred_mean-pred_rest);
    c0 = mu_c0 + sqrt(tau2 / n)*randn(1);
        
    %% update tau2 
    a_tau = a_t + n/2;
    b_tau = b_t + 0.5*norm(obs-pred_mean-pred_rest-c0)^2;
    tau2 = 1/gamrnd(a_tau, 1./b_tau, 1);

    %% sample astar
    o = draw_phi_tau(alpha_grid, beta, omega, b_vphi);
    astar = randsample(alpha_grid, 1, true, o.scores);
    score_store{sweep} = o.scores;
        
    %% sample (phi, varphi)
    o = draw_phi_tau(astar, beta, omega, b_vphi);
    phi_val = o.phi; 
    varphi_val = o.varphi;
    tau_r = varphi_val * phi_val;
                
    %% update rank specific params
    for j = 1:d
        m_set = cellfun(@sum,cellfun(@ismember, modes, num2cell(j*ones(1,M)),'UniformOutput',false));
        m_inds = find(m_set);
        idx = setdiff(1:M,m_inds);
        pred_rest = zeros(n,1);
        for m_n = 1:length(idx)
            pred_rest = pred_rest+getmean(Xt{idx(m_n)},beta(:,modes{idx(m_n)}),rank,sz(modes{idx(m_n)}));
        end
        for r = 1:rank
            H = zeros(n, sz(j));
            tens_mu_r = zeros(n,1);
            for m = 1:length(m_inds)
                m_t = m_inds(m);
                tens_mu_r = tens_mu_r + getmean(Xt{m_t},beta(:,modes{m_t}),rank,sz(modes{m_t}),r);
                betj = getouter(beta(r,modes{m_t}),find(modes{m_t}==j));
                H = H + reshape(t2m(Xt{m_t},[1,find(modes{m_t}==j)+1])*betj,[n,sz(j)]); %N x I_d
            end
            mm = (obs-c0-pred_mean-pred_rest-tens_mu_r);
            K = H'*H./tau2 + diag(1./(tau_r(r)*omega{j}(r,:)));  % I_d x I_d

            % update betas
            bet_mu_jr = K\((H'/tau2)*mm);
            beta{r,j} = bet_mu_jr+chol(inv(K))*randn(sz(j),1);

            % update lambda.jr
            lambda(r,j) = gamrnd(a_lam(r)+sz(j), ...
                (b_lam(r)+sum(abs(beta{r,j}))/sqrt(tau_r(r)))^-1, 1);

            % update omega.jr
            for k = 1:sz(j)
                omega{j}(r,k) = gigrnd(1/2, lambda(r,j)^2, beta{r,j}(k)^2/tau_r(r));
            end
        end
    end
    beta_store{sweep} = beta;
        
    %% store params
    tau2_store{sweep} = tau2;
    c0_store{sweep} = c0;
    gam_store{sweep} = gam;
    alpha_store{sweep} = astar; % not intercept
    phi_store{sweep} = phi_val;
    varphi_store{sweep} = varphi_val;
    omega_store{sweep} = omega;
    lambda_store{sweep} = lambda;
    hyppar_store{sweep} = {a_lam, b_lam};
end
    
    tt = toc(tt);
    disp(['Time out: ', num2str(tt)])
            
    %% finalize ####    
    out = struct('nsweep', nsweep, 'obs',  obs, 'a_t',  a_t, 'b_t',  b_t, 'tau2_store',  tau2_store,...
        'c0_store', c0_store, 'gam_store', gam_store, ...
        'alpha_store', alpha_store, 'beta_store', beta_store, ...
        'phi_store', phi_store, 'varphi_store', varphi_store,...
        'omega_store',  omega_store, 'lambda_store',  lambda_store, ...
        'hyppar_store',  hyppar_store, 'score_store',  score_store, 'time',  tt);
end

    
%% Aux functions ####
% function o = mfun(z, p)
% o = gammaln(z(1)+p) - gammaln(z(1)) + z(1)*log(z(2)*z(1)) - (z(1)+p)*log(z(2)*z(1));
% end

function out = draw_phi_tau(alpha, beta, omega, b_vphi)

M = 20;   

len_alpha = length(alpha);
rank = size(beta,1);
d = size(beta,2);
p = zeros(1,d);
for j=1:d
    p(j) = length(beta{1,j});
end
m.phialpha = repmat(alpha(1), rank, 1);
m.phia0 = sum(m.phialpha);
m.avphi = m.phia0;
% assumes b_vphi const (use: alpha 1 / R)

Cr = zeros(rank, d);
for r=1:rank
    for j=1:d
        Cr(r,j) = sum((beta{r,j}.^2)./omega{j}(r,:)'); % 1
    end
end

if(len_alpha > 1)
    phi_val = zeros(M*len_alpha, rank);
    varphi_val = zeros(M*len_alpha, 1);
    Cstat_val = zeros(M*len_alpha, rank);
    scores = zeros(len_alpha,M*len_alpha);

    % get reference set
    for jj = 1:len_alpha  
        m.phialpha = repmat(alpha(jj), rank,1);
        m.phia0 = sum(m.phialpha);
        m.avphi = m.phia0;

        % draw phi
        Cr1 = sum(Cr,2);
        for r = 1:rank
            for j=1:M
                phi.a(r,j) = gigrnd(m.phialpha(r)-sum(p)/2,2*b_vphi,Cr1(r));
            end
        end
        phi.a = phi.a*diag(1./sum(phi.a,1)); % RxM

        % draw varphi ##colSums(Cr / t(replicate(d, z)))
        Cr2 = (Cr1./phi.a)';
        for j=1:M
            varphi.a(j) = gigrnd(m.avphi-rank*sum(p)/2, 2*b_vphi, sum(Cr2(j,:)));
        end
        phi_val((jj-1)*M+1:jj*M, :) = phi.a';
        varphi_val((jj-1)*M+1:jj*M) = varphi.a;
        Cstat_val((jj-1)*M+1:jj*M, :) = Cr2;
    end
    for i =1:len_alpha
        scores(i,:) = score_fn(repmat(alpha(i),rank,1), phi_val, b_vphi, varphi_val, Cstat_val, p);
    end
    lmax = max(scores,[],'all');
    scores = mean(exp(scores-lmax),2);
else
    % draw phi
    Cr1 = sum(Cr,2);
    phi_val = zeros(1,rank);
    for r = 1:rank
        phi_val(r) = gigrnd(m.phialpha(r)-sum(p)/2, 2*b_vphi, Cr1(r));
    end
    phi_val = phi_val ./ sum(phi_val);

    % draw varphi
    Cr2 = Cr1 ./ phi_val';
    varphi_val = gigrnd(m.avphi-rank*sum(p)/2, 2*b_vphi, sum(Cr2));
    scores = NaN;
end
out.phi = phi_val; 
out.varphi = varphi_val;
out.scores = scores;
end

function o = score_fn(phi_alpha, phi_s, b_vphi, varphi_s, Cstat, p)

ldir = gammaln(sum(phi_alpha))-sum(gammaln(phi_alpha))+sum(log(phi_s)*diag(phi_alpha-1),2);
lvarphi = log(gampdf(varphi_s, sum(phi_alpha), 1./b_vphi));

dnorm_log = -sum(Cstat, 2)./(2 * varphi_s) - (sum(p)/2) * sum(log(diag(varphi_s) * phi_s),2);
o = dnorm_log + ldir + lvarphi;
end
% rmse = (a, b) return(sqrt(mean((a - b)^2)))

% logsum = function(lx) return(max(lx) + log(sum(exp(lx - max(lx)))))