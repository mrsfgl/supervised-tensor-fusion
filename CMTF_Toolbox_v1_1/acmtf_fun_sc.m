function [f,g] = acmtf_fun_sc(x, Z, R, Znormsqr, beta_cp, beta_pca, alpha)
% ACMTF_FUN_SC computes the function value and the gradient for soft coupled
% matrix and tensor factorization, where the matrix model is a matrix factorization
% model and the tensor model is a CANDECOMP/PARAFAC (CP) model (with the option 
% of adding sparsity penalties on the weights of components through the use
% of parameters beta_cp and beta_pca).
% 
% [f,g] = acmtf_fun_sc(x, Z, R, Znormsqr, beta_cp, beta_pca)
%
% Input:   x: a vector of concatenated vectorized factor matrices as well as 
%             the norms for each factor; x = acmtf_struct_to_vec(G), where 
%             G.fac has the factor matrices and G.norms has the norms of each 
%             component.
%          Z: a structure with object, modes, size fields storing the
%             coupled data sets (See cmtf_check).
%          R: number of components
%          Znormsqr: a cell array with squared Frobenius norm of each Z.object
%          beta_cp : sparsity parameter on the weights of rank-one tensors
%                    in the CP part of a CMTF model.
%          beta_pca: sparsity parameter on the weights of rank-one matrices
%                    in the PCA part of a CMTF model.
%          
% Output:  f: function value of the combined objective function.
%          g: a vector corresponding to the gradient.
%
% See also ACMTF_OPT, ACMTF_FG, ACMTF_VEC_TO_STRUCT, ACMTF_STRUCT_TO_VEC,
% SCP_FG, SCP_WFG, SPCA_FG, SPCA_WFG, CMTF_CHECK.
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

%% Convert the input vector into a struct of cell array of factor matrices and 
% norms, i.e., A.fac and A.norms
A = acmtf_vec_to_struct_sc(x,Z,R); 

%% Compute the function and gradient values
[f,G] = acmtf_fg_wsc(Z,A,Znormsqr,beta_cp,beta_pca,alpha);

theta = 1;
N = length(G.fac);
R = size(G.fac{1},2);
% add norm constraints to gradient
for i=1:N
    for r=1:R         
         G.fac{i}(:,r) = G.fac{i}(:,r) + theta*(A.fac{i}(:,r)-(A.fac{i}(:,r)/norm(A.fac{i}(:,r))));
    end
end

% add norm constraints to f
for i=1:N
    for r=1:R        
        f = f + 0.5* theta * (norm(A.fac{i}(:,r))-1)^2;
    end
end

% Vectorize the cell array of factor matrices
g = acmtf_struct_to_vec(G); 

