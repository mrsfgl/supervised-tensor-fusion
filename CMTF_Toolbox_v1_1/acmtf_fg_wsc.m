function [f,G] = acmtf_fg_wsc(Z,A,Znormsqr, beta_cp, beta_pca, alpha)
% ACMTF_FG_WSC Function and gradient of soft coupled matrix and tensor  
% factorization, where the coupled model is formulated, for instance, for 
% a third-order tensor and a matrix coupled in the first mode as 
% f = 0.5*||Z.object{1}-[|\Lambda; A_1, B, C|]||^2 
%     + 0.5* ||Z.object{2} -A_2\SigmaD'||^2+0.5*alpha*||A_1-A_2||_F^2
%     + 0.5*beta_cp*|\Lambda|_1 + 0.5*beta_pca *|diag(\Sigma)|_1 + P, 
% where P indicates quadratic penalty terms to normalize the columns of 
% factor matrices to unit norm.
%
% [f,G] = acmtf_fg_wsc(Z, A, Znormsqr, beta_cp, beta_pca, alpha) 
%
% Input:  Z: a struct with object, modes, size, miss fields storing the 
%            coupled data sets (See cmtf_check)
%         A: a struct with fac (a cell array corresponding to factor 
%            matrices) and norms (a cell array corresponding to the weights
%            of components in each data set) fields. 
%         Znormsqr: a cell array with squared Frobenius norm of each Z.object
%         beta_cp : sparsity parameter on the weights of rank-one tensors
%                   in the CP part of a CMTF model.
%         beta_pca: sparsity parameter on the weights of rank-one matrices
%                   in the PCA part of a CMTF model.
%         alpha: Coupling strength parameter.
% 
% Output: f: function value
%         G: a struct with fac (i.e., G.fac corresponding to the part of the 
%            gradient for the factor matrices) and norms (i.e., G.norms 
%            corresponding to the part of the gradient for the weights of
%            the components in each data set) fields.
%
% See also ACMTF_FG, ACMTF_FUN, SPCA_FG, SPCA_WFG, SCP_FG, SCP_WFG
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


if ~isstruct(A)
    error('A must be a structure with fields: fac and norms, each being a cell array');
end

P = numel(Z.object);

fp   = cell(P,1);
Gp   = cell(P,1);
n_modes = 0;
modes = [];
for p = 1:P   
    B = [A.fac(n_modes+(1:length(Z.modes{p}))); A.norms{p}];
    n_modes = n_modes+length(Z.modes{p});
    if length(size(Z.object{p}))>=3
        % Tensor
        if isfield(Z,'miss') && ~isempty(Z.miss{p})            
            [fp{p},Gp{p}] = scp_wfg(Z.object{p}, Z.miss{p}, B, Znormsqr{p}, beta_cp);        
        else
            [fp{p},Gp{p}] = scp_fg(Z.object{p}, B, Znormsqr{p}, beta_cp);        
        end
    elseif length(size(Z.object{p}))==2
        % Matrix
        if isfield(Z,'miss') && ~isempty(Z.miss{p})
            [fp{p},Gp{p}] = spca_wfg(Z.object{p}, Z.miss{p}, B, Znormsqr{p}, beta_pca);                        
        else
            [fp{p},Gp{p}] = spca_fg(Z.object{p}, B, Znormsqr{p}, beta_pca);                         
        end
    end
    G.norms(p) = Gp{p}(end);  
    modes = [modes,Z.modes{p}];
end

%% Compute overall gradient
n_modes = cellfun(@length, Z.modes);
n_modes = cumsum(n_modes);
for n = 1:n_modes(end)
    G.fac{n} = zeros(size(A.fac{n}));
end
for p = 1:P
    for i = 1:length(Z.modes{p})
        j = Z.modes{p}(i);
        id = find(modes==j);
        curr_m = n_modes(p-1)+i;
        id = setdiff(id, curr_m);
        for k=1:length(id)
            G.fac{curr_m} = G.fac{curr_m} + alpha*A.fac{curr_m};
            G.fac{id(k)} = G.fac{id(k)} - alpha*A.fac{curr_m};
        end
        G.fac{curr_m} = G.fac{curr_m} + Gp{p}{i};
    end    
end

%% Compute overall function value
f = sum(cell2mat(fp));

