function out = bater_multimodal_concat(z_train, x_train, y_train, nsweep, rank) 
%%
%

n = length(y_train);
M = length(x_train);
for m=1:M
    d(m) = ndims(x_train{m})-1;
    p{m} = size(x_train{m}, 2:d(m)+1);
end
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
        ones(1,d(m))]))/diag(sx);
end
%% MCMC Setup
ZZ = z_train'*z_train;

%% Initialize 
% Hyperparameters
a_lam = repmat(3, 1, rank);
b_lam = cell(M,1);
for m = 1:M
    b_lam{m} = a_lam.^(1/(2*d(m)));
end
phi_alpha = repmat(1/rank, 1, rank);

for m = 1:M
    b_vphi{m} = phi_alpha(1)*rank^(1/4);
end

c0 = 0;
s0 = 1;
a_t = 2.5/2;
b_t = 2.5/2 * s0^2;
tau2 = 1 / gamrnd(a_t, 1./b_t, 1); % initialize tau2

lambda = cell(M,1);
omega = cell(M,1);
for m = 1:M
    lambda{m} = gamrnd(a_lam(1), 1./b_lam{m}(1), rank, d(m)); % initialize lambda with size rank x d(m)
end
for m = 1:M
    omega{m} = cell(d(m),1);
    for j = 1:d(m)
        omega{m}{j} = exprnd(0.5*(a_lam(1)/b_lam{m}(1)), rank, p{m}(j));% initialize omega
    end
end
beta = cell(M,1);
for m = 1:M
    beta{m} = cell(rank,d(m));
    for j = 1:d(m)
        for r = 1:rank
            beta{m}{r,j} = randn(p{m}(j),1); % initialize beta
        end
    end
end
alpha_grid = cell(M,1);
for m=1:M
    alpha_grid{m} = linspace(rank^(-d(m)), rank^(-0.1), len_alpha);  % grid of alpha values
end

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
        tens_mean{m} = getmean(Xt{m}, beta{m}, rank, p{m}, []);
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
    for m = 1:M
        o = draw_phi_tau(alpha_grid{m}, beta{m}, omega{m}, b_vphi{m});
        astar{m} = randsample(alpha_grid{m}, 1, true, o.scores);
        score_store{sweep}{m} = o.scores;
    end
        
    %% sample (phi, varphi)
    for m = 1:M
        o = draw_phi_tau(astar{m}, beta{m}, omega{m}, b_vphi{m});
        phi_val{m} = o.phi; 
        varphi_val{m} = o.varphi;
        tau_r{m} = varphi_val{m} * phi_val{m};
        a_vphi{m} = sum(repmat(astar{m}, rank,1)); 
    end
                
    %% update rank specific params
    for m = 1:M
        pred_rest = zeros(n,1);
        idx = setdiff(1:M,m);
        for m_n = idx
            pred_rest = pred_rest + getmean(Xt{m_n}, beta{m_n},rank,p{m_n});
        end
        for r = 1:rank
            tens_mu_r = getmean(Xt{m}, beta{m}, rank, p{m}, r);
            for j = 1:d(m)
                betj = getouter(beta{m}(r,:), j);
                H = reshape(t2m(Xt{m},[1,j+1])*betj,[n,p{m}(j)]); %NxI_d
                K = H'*H./tau2 + diag(1./(tau_r{m}(r)*omega{m}{j}(r,:)));  % I_dxI_d

                % update betas
                mm = (obs-c0-pred_mean-pred_rest-tens_mu_r);
                bet_mu_jr = K\((H'/tau2)*mm);
                beta{m}{r,j} = bet_mu_jr+chol(inv(K))*randn(p{m}(j),1);

                % update lambda.jr
                lambda{m}(r,j) = gamrnd(a_lam(r)+p{m}(j), ...
                    (b_lam{m}(r)+sum(abs(beta{m}{r,j}))/sqrt(tau_r{m}(r)))^-1, 1);

                % update omega.jr
                for k = 1:p{m}(j)
                    omega{m}{j}(r,k) = gigrnd(1/2, lambda{m}(r,j)^2, beta{m}{r,j}(k)^2/tau_r{m}(r));
                end
            end
        end
        beta_store{sweep}{m} = beta{m};
    end
        
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
%         m.phia0 = sum(m.phialpha);
        m_avphi = alpha(jj)*rank; %m.phia0;

        % draw phi
        Cr1 = sum(Cr,2);
        for r =1:rank
            for j=1:M
                phi.a(r,j) = gigrnd(m.phialpha(r)-sum(p)/2,2*b_vphi,Cr1(r));
            end
        end
        phi.a = phi.a*diag(1./sum(phi.a,1)); % RxM

        % draw varphi ##colSums(Cr / t(replicate(d, z)))
        Cr2 = (Cr1./phi.a)';
        for j=1:M
            varphi.a(j) = gigrnd(m_avphi-rank*sum(p)/2, 2*b_vphi, sum(Cr2(j,:)));
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
    varphi_val = gigrnd(m_avphi-rank*sum(p)/2, 2*b_vphi, sum(Cr2));
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