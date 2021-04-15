function [f, X] = admm_loop(Z, X, Znormsqr, alpha, beta, eta, theta)
% SCP_FG Computes function and gradient of the CP function. SCP_FG also 
% has the option of imposing sparsity on the weights of rank-one components.
%
%   [f,X] = scp_fg(Z, A, Znormsqr, beta) 
%           
% Input:  Z: an N-way tensor 
%         A: a cell array of length N+1. The last cell corresponds to 
%           the weights of the components. 
%         Znormsqr: norm of Z. 
%         beta: sparsity penalty parameter on the weights of rank-one tensors.
%
% Output: f: function value computed as 
%               f = 0.5*||Z - ktensor(A{end}, A(1:end-1))||^2  
%                   + 0.5*beta*|A{end}|_1 
%            where l1-penalty is replaced with a smooth approximation.
%         X: a cell array of length N+1.
%
% See also SCP_WFG, ACMTF_FG
%
% This is the MATLAB CMTF Toolbox, 2013.
% References: 
%    - (CMTF) E. Acar, T. G. Kolda, and D. M. Dunlavy, All-at-once Optimization for Coupled
%      Matrix and Tensor Factorizations, KDD Workshop on Mining and Learning
%      with Graphs, 2011 (arXiv:1105.3422v1)
%    - (ACMTF)E. Acar, A. J. Lawaetz, M. A. Rasmussen,and R. Bro, Structure-Revealing Data 
%      Fusion Model with Applications in Metabolomics, IEEE EMBC, pages 6023-6026, 2013.
%    - (ACMTF)E. Acar,  E. E. Papalexakis, G. Gurdeniz, M. Rasmussen, A. J. Lawaetz, M. Nilsson, and R. Bro, 
%      Structure-Revealing Data Fusion, BMC Bioinformatics, 15: 239, 2014.        
%


P = length(X);
%% Initialize variables.
Y = X;
for p = 1:P
    for i = 1:length(X{p})
        Omega = zeros(size(X{p}{i}));
    end
end

iter = 1;
while iter<max_iter
    %% Update of variables X.
    [X, f_1] = update_X(Z, X, Y, Omega, beta(1), alpha, eta);

    %% f1
    f_1 = f_1+Znormsqr;

    %% Update of variables Y
    [Y, f_2] = update_Y(X, Omega, eta, theta, beta(2));

    %% Update the Lagrange multipliers.
    for p = 1:P
        for i = 1:length(X{p})
            Omega{p}{i} = Omega{p}{i} - Y{p}{i} + X{p}{i};
        end
    end
    
    %% Ending conditions
    iter = iter+1;
end

f = f_1+f_2;
end

function [X, obj_val] = update_X(Z, X, Y, Omega, beta_res, alpha, eta)

P = length(X);
R = size(X{1}{1},2);
modes = cell2mat(Z.modes);

%% Initialize Upsilon and Gamma for inverse calculations
Upsilon = cell(P,1);
for p = 1:P
    Upsilon{p} = cell(1,length(X{p}));
    for i = 1:length(Z.modes{p})
        Upsilon{p}{i} = X{p}{i}'*X{p}{i};
    end
    Upsilon{p}{i+1} = X{p}{end}*X{p}{end}';
end

Gamma = cell(P,1);
obj_val = 0;
for p = 1:P
    Gamma{p} = ones(R,R);
    for i = 2:length(Z.modes{p})
        Gamma{p} = Gamma{p} .* Upsilon{p}{i};
    end
    Gamma{p} = Gamma{p} .* Upsilon{p}{i+1};
    obj_val = obj_val + beta_res*trace(Gamma{p}.*Upsilon{p}{1});
end

%% Update all factor matrices and component weights.
n_modes = cumsum(cellfun(@length, Z.modes));
for p = 1:P
    %% Update the first factor and compute the objective.
    U = khatrirao(mttkrp(Z.object{p},X{p}(1:end-1),1), X{p}{end}');
    obj_val = obj_val - 2*beta_res*trace(U'*X{p}{1});
    
    [U, w, val] = check_sc(U, X, p, 1, n_modes, modes, eta(1,p), alpha);
    obj_val = obj_val + val;
    
    inv_gamma = (beta_res*Gamma{p}+w*eye(R))^-1;
    V = Y{p}{1}-Omega{p}{1};
    % Compute the contribution to the objective
    obj_val = obj_val + eta(1,p)/2*norm(X{p}{1}-V,'fro')^2;
    % Update factors
    X{p}{1} = (U+(eta(1,p)/2)*V)*inv_gamma;
    
    %% Update factors with order >2
    for i=2:length(Z.modes{p})
        %% Account for the soft coupling with any coupled modalities.
        U = beta_res*khatrirao(mttkrp(Z.object{p},X{p}(1:end-1),i), X{p}{end}');

        [U, w, val] = check_sc(U, X, p, i, n_modes, modes, eta(i,p), alpha);
        obj_val = obj_val + val;

        %% Compute inverse and solve the linear system.
        inv_gamma = (beta_res*Gamma{p}+w*eye(R))^-1;

        V = Y{p}{i}-Omega{p}{i};
        % Compute the contribution to the objective
        obj_val = obj_val + eta(1,p)/2*norm(X{p}{i}-V,'fro')^2;
        % Update factors
        X{p}{i} = (U+(eta(1,p)/2)*V)*inv_gamma;
        
        %% Update the Gamma and Upsilon for inverse calculations.
        Gamma{p} = Gamma{p}./Upsilon{p}{i+1};
        Upsilon{p}{i} = X{p}{i}'*X{p}{i};
        Gamma{p} = Gamma{p}.*Upsilon{p}{i};
    end
    %% Update weights of rank one components.
    inv_gamma = (beta_res*Gamma{p}+eta(2,p)/2*eye(R))^-1;
    U = mttkrp(Z.object{p},X{p}(1:end-1),i+1);
    V = Y{p}{i+1}-Omega{p}{i+1};
    % Compute the contribution to the objective
    obj_val = obj_val + eta(2,p)*norm(X{p}{i+1}-V,'fro')^2;
    % Update the weights
    X{p}{i+1} = inv_gamma*(beta_res*U+eta(2,p)/2*V);
end

end

function [Y, obj_val] = update_Y(X, Omega, eta, theta, beta)

obj_val = 0;
Y = X;
for p=1:P
    %% Update the variables for 2-1 norm.
    for i = 1:length(Y{p})-1
        [Y{p}{i}, val] = update_21(X{p}{i}+Omega{p}{i}, eta(1,p), theta);
        obj_val = obj_val + val;
    end
    %% Update the variables for sparsity penalty on component weights.
    [Y{p}{i+1}, val] = update_st(X{p}{i+1}+Omega{p}{i+1}, eta(2,p), beta);
    obj_val = obj_val + val;
end
end

function [U, w, obj_val] = check_sc(U, X, p, i, n_modes, modes, eta, alpha)

n = n_modes(p-1)+i;
j = modes(n);
id = find(modes==j);
id = setdiff(id, n);
w = eta;
obj_val = 0;
for k=1:length(id)
    p_c = find(n_modes>=id(k),1,'first');
    w = w + alpha;
    V = X{p_c}{id(k)-n_modes(p_c-1)};
    U = U + alpha*V;
    obj_val = obj_val + alpha/2*norm(X{p}{i}-V,'fro')^2;
end
end

function [Y, obj_val] = update_21(V, eta, theta)

R = size(V,2);
Y = zeros(size(V));
obj_val = 0;
for r = 1:R
    n = norm(V(:,r));
    obj_val = obj_val + theta*(n-1)^2;
    coeff = (2*theta+eta*n)/((2*theta+eta)*n);
    Y(:,r) = coeff*V(:,r);
    obj_val = obj_val + eta*norm(Y(:,r)-V(:,r))^2;
end
end

function [Y, obj_val] = update_st(V, eta, theta)

Y = zeros(size(V));
temp = (abs(V)-theta/eta);
mask = temp>0;
Y(mask) = temp(mask)*sign(V(mask));

obj_val = norm(Y(:)-V(:), 1);
end
