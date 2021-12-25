function out = bater(z_train, x_train, y_train, nsweep, rank) 
%%
%

n = length(y_train);
p = size(x_train, 2:ndims(x_train));
d = length(p);
len_alpha = 10;
pgamma = size(z_train, 2); %ztrain\in R^{n x pgamma}

%% Standardize
my = mean(y_train,1);
sy = std(y_train,[],1);
obs = (y_train-my)/sy;

sx = max(x_train,[],1)-min(x_train,[],1);
sx(sx==0) = 1;
Xt = (x_train-repmat(mean(x_train,1),[size(x_train,1),...
    ones(1,ndims(sx)-1)]))./repmat(sx,[size(x_train,1),ones(1,ndims(sx)-1)]);

%% MCMC Setup
ZZ = z_train'*z_train;

%% Initialize 
% Hyperparameters
a_lam = repmat(3, 1, rank);
b_lam = a_lam.^(1/(2*d));
phi_alpha = repmat(1/rank, 1, rank);

a_vphi = sum(phi_alpha);
b_vphi = phi_alpha(1)*rank^(1/4);

c0 = 0;
s0 = 1;
a_t = 2.5/2;
b_t = 2.5/2 * s0^2;
tau2 = 1 / gamrnd(a_t, 1./b_t, 1); % initialize tau2

lambda = gamrnd(a_lam(1), 1./b_lam(1), rank, d); % initialize lambda with size rank x d
omega = cell(d,1);
for m = 1:d
    omega{m} = exprnd(0.5*(a_lam(1)/b_lam(1)), rank, p(d));% initialize omega
end
beta = cell(rank,d);
for r = 1:rank
    for j = 1:d
        beta{r, j} = randn(p(j),1); % initialize beta
    end
end

alpha_grid = linspace(r^(-d), r^(-0.1), len_alpha);  % grid of alpha values

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

for sweep = 1:nsweep
    tens_mean = getmean(Xt, beta, rank, p, []);
        
    %% update (a_lam, b_lam)
%     Cjr = zeros(rank,d);
%     for i=1:rank
%         for j=1:d
%             Cjr(r,d) = sum(abs(beta{r,j}))/sqrt(tau_r(r)); 
%         end
%     end
%     for l = 1:size(par_grid,1)
%         par.wt(l, :) = sum(mfun(par_grid(l,:))+Cjr,2);
%     end
%     par.wt = exp(par.wt-log(sum(exp(par.wt),2)));
%     ixx = zeros(1,size(par_grid,1));
%     for k = 1:size(par_grid, 1)
%         ixx(k) = randsample(size(par_grid, 1), 1, true, par.wt(k,:)); %%%% Ask Sumegha
%     end
%     for rr = 1:rank
%         a_lam(rr) = par_grid(ixx(rr),1);
%         b_lam(rr) = par_grid(ixx(rr),2) * a_lam(rr);
%     end
        
    %% update gamma (scalar predictor coefficients)
    Sig_g = eye(pgamma) + ZZ / tau2;%eye(pgamma)-z_train*z_train'/(tau2+z_train'*z_train);
    mu_g = Sig_g \ (z_train'*(obs-c0-tens_mean)/tau2);
    gam = mu_g + chol(Sig_g) * randn(pgamma,1);
        
    %% update alpha (intercept) 
    pred_mean = z_train * gam;
    mu_c0 = mean(obs-pred_mean-tens_mean);
    c0 = mu_c0 + sqrt(tau2 / n)*randn(1);
        
    %% update tau2 
    a_tau = a_t + n/2;
    b_tau = b_t + 0.5*norm(obs-pred_mean-tens_mean-c0)^2;
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
    phi_alpha = repmat(astar, rank,1); 
    phi_a0 = sum(phi_alpha); 
    a_vphi = phi_a0;
                
    %% update rank specific params
    for r = 1:rank
        for j = 1:d
            tens_mu_r = getmean(Xt, beta, rank, p, r);

            betj = getouter(beta(r,:), j);
            H = reshape(t2m(Xt,[1,j+1])*betj,[n,p(j)]); %NxI_d
            K = H'*H./tau2 + diag(1./(tau_r(r)*omega{j}(r,:)));  % I_dxI_d

            % update betas
            mm = (obs-c0-pred_mean-tens_mu_r);
            bet_mu_jr = K\((H'/tau2)*mm);
            beta{r,j} = bet_mu_jr+chol(inv(K))*randn(p(j),1);

            % update lambda.jr
            lambda(r,j) = gamrnd(a_lam(r)+p(j), ...
                (b_lam(r)+sum(abs(beta{r,j}))/sqrt(tau_r(r)))^-1, 1);

            % update omega.jr
            for k = 1:p(j)
                omega{j}(r,k) = gigrnd(1/2, lambda(r,j)^2, beta{r,j}(k)^2/tau_r(r));
            end
            %omega[r,j,] = sapply(1:p, function(kk){a = lambda[r,j]^2; b = beta[[r]][kk,j]^2 / tau_r[r]; map = besselK(sqrt(a*b),0.5 + 1) / besselK(sqrt(a*b), 0.5) * sqrt(b / a); return(map)})
        

        beta_store{sweep} = beta;
        end
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
    out = struct('nsweep', nsweep, 'rank',  rank, 'p',  p, 'd',  d, ...
        'alpha_grid',  alpha_grid, 'my',  my, 'sy',  sy, ...
        'Xt',  Xt, 'obs',  obs, 'a_t',  a_t, 'b_t',  b_t, 'tau2_store',  tau2_store,...
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
        for r =1:rank
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