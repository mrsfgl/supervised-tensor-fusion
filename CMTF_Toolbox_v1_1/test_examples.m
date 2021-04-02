%% The MATLAB CMTF Toolbox v1.1, 2014
% The MATLAB CMTF Toolbox has two different versions of the Coupled Matrix and Tensor Factorization approach used to jointly analyze datasets of different orders: 
% (i) CMTF [1] and (ii) ACMTF [2, 5]. First-order unconstrained optimization is used to fit both versions. The MATLAB CMTF Toolbox has the functions necessary to
% compute function values and gradients for CMTF and ACMTF. For the optimization routines, it uses the Poblano Toolbox [3]. The Tensor Toolbox [4] is also needed to
% run the functions in the Matlab CMTF Toolbox. This page illustrates some example scripts, e.g., TESTER_CMTF, TESTER_ACMTF, TESTER_CMTF_MISSING, TESTER_ACMTF_MISSING
% showing the use of CMTF and ACMTF. For more details, explore CMTF_OPT and ACMTF_OPT. 
%
%% What is new in Version 1.1?
% # Compatibility with sptensor is added to make CMTF_OPT and ACMTF_OPT work with tensors in sptensor form.
% # TESTER_CMTF and TESTER_ACMTF have been modified to have the option of generating data sets in dense or sparse tensor format.
% # TESTER_CMTF_MISSING and TESTER_ACMTF_MISSING functions have been added to demonstrate the use of CMTF_OPT and ACMTF_OPT with data sets with missing entries.
% # CREATE_COUPLED_SPARSE function has been added to generate coupled sparse data sets using sparse factor matrices.
% # For smooth approximation of the l1-term in SCP_FG, SCP_WFG, SPCA_FG, SPCA_WFG, eps is set to 1e-8.
%

%% CMTF (Coupled Matrix and Tensor Factorizations) using first-order optimization
% Coupled Matrix and Tensor Factorizations model higher-order tensors using CANDECOMP/PARAFAC (CP) models and factorizes matrices jointly.
% TESTER_CMTF is an example script showing how to use CMTF.

% Example1:
% Generate a third-order data set coupled with a matrix and fit a CMTF model. 
data = tester_cmtf;

% Check how well the true factors used to generate coupled data sets, i.e., data.Factrue{i}, match with the factor matrices extracted using CMTF,
% i.e., data.Fac{i}. For example, for the first mode, i=1:
corr(data.Fac{1}, data.Factrue{1})

% Example2:
% Generate a third-order tensor coupled with a matrix in the first mode and coupled with another matrix in the second mode. Three components are used
% to generate data.
data = tester_cmtf('modes',{[1 2 3], [1 4], [2 5]}, 'size', [30 10 20 40 50], 'lambdas',{[1 1 1],[1 1 1], [1 1 1]}, 'flag_sparse', [0 0 0]);

% Check how well the true factors used to generate coupled data sets, i.e., data.Factrue{i}, match with the factor matrices extracted using CMTF,
% i.e., data.Fac{i}. For example, for the first mode, i=1:
corr(data.Fac{1}, data.Factrue{1})

% Example3:
% Generate a sparse third-order tensor coupled with a matrix stored in dense format and fit a CMTF model
data = tester_cmtf('modes',{[1 2 3],[1 4]}, 'size', [100 100 100 100],'flag_sparse',[1 0]);

% Check how well the true factors used to generate coupled data sets, i.e., data.Factrue{i}, match with the factor matrices extracted using CMTF,
% i.e., data.Fac{i}. For example, for the first mode, i=1:
corr(data.Fac{1}, data.Factrue{1})

% Example4:
% Generate a sparse third-order tensor coupled with a sparse matrix and fit a CMTF model
data = tester_cmtf('modes',{[1 2 3],[1 4]}, 'size', [100 100 100 100],'flag_sparse',[1 1]);

% Check how well the true factors used to generate coupled data sets, i.e., data.Factrue{i}, match with the factor matrices extracted using CMTF,
% i.e., data.Fac{i}. For example, for the first mode, i=1:
corr(data.Fac{1}, data.Factrue{1})

%% ACMTF (Coupled Matrix and Tensor Factorization) using first-order optimization with the option of imposing sparsity penalties on the component weights
% Coupled Matrix and Tensor Factorizations model higher-order tensors using CANDECOMP/PARAFAC models and factorizes
% matrices jointly. Unlike CMTF, ACMTF enables the option of imposing sparsity penalties on the weights of components
% in order to identify shared/unshared components in coupled data sets. TESTER_ACMTF is an example script showing how to use ACMTF.

% Example 1
% Generate a third-order tensor coupled with a matrix with one shared component and one unshared component in each data set, and fit a CMTF model with penalties 
% on the component weights.
data = tester_acmtf('R',3,'size',[30 20 10 40], 'lambdas',{[1 0 1],[0 1 1]}, 'modes',{[1 2 3],[1 4]},'beta_cp',0.001, 'beta_pca',0.001);

% Check how well extracted factors, i.e., Fac1 and Fac2, match with the original ones, data.Atrue. 
Fac1 = normalize(data.Zhat{1});
Fac2 = normalize(data.Zhat{2});
for i=1:3
    corr(Fac1{i},data.Atrue{i})    
end
corr(Fac2{2},data.Atrue{4})

% Check whether the weights reveal shared/unshared components
data.lambda_rec

% Example 2
% Generate a sparse third-order tensor coupled with a sparse matrix with one shared component and one unshared component in each data set, and fit a CMTF model with 
% penalties on the component weights.
data = tester_acmtf('R',3,'size',[30 20 10 40], 'lambdas',{[1 0 1],[0 1 1]}, 'modes',{[1 2 3],[1 4]},'beta_cp',0.001, 'beta_pca',0.001,'flag_sparse',[1 1]);

% Check how well extracted factors, i.e., Fac1 and Fac2, match with the original ones, data.Atrue. 
Fac1 = normalize(data.Zhat{1});
Fac2 = normalize(data.Zhat{2});
for i=1:3
    corr(Fac1{i},data.Atrue{i})    
end
corr(Fac2{2},data.Atrue{4})

% Check whether the weights reveal shared/unshared components
data.lambda_rec

%% Joint Analysis of Incomplete Data Sets using CMTF/ACMTF
% TESTER_CMTF_MISSING and TESTER_ACMTF_MISSING are example scripts showing
% how to use CMTF and ACMTF with incomplete data sets.

% Example 1
% Generate a third-order tensor (with 50% of its entries missing) coupled with a matrix in the first mode.
data = tester_cmtf_missing('size',[20 30 40 50], 'modes', {[1 2 3], [1 4]}, 'R', 3, 'M', [0.5 0]);

% Compute the error between the true values of missing entries and the estimated values.
trueval = data.Xorig{1}(find(data.W{1}==0));
Z       = full(ktensor(data.Fac(1:3)));
estval  = Z(find(data.W{1}==0));
plot(trueval,estval,'*');xlabel('True Values');ylabel('Estimated Values'); title('Missing Data Estimation');
err     = norm(estval - trueval)/length(estval);

% Example 2
% Generate a third-order tensor (with 80% of its entries missing) coupled with a matrix (with 50% of its entries missing) in the first mode. Both
% data sets are stored in sptensor form. Data sets have one shared and one unshared components.
data = tester_acmtf_missing('size',[20 30 40 50], 'modes', {[1 2 3], [1 4]}, 'R', 3, 'M', [0.8 0.5],'flag_sparse',[1 1],'lambdas',{[1 0 1],[0 1 1]},'beta_cp',0.001, 'beta_pca',0.001);

% Compute the error between the true values of missing entries and the estimated values.
trueval = data.Xorig{1}(find(data.W{1}==0));
Z       = full(data.Zhat{1});
estval  = Z(find(data.W{1}==0));
plot(trueval,estval,'*');xlabel('True Values');ylabel('Estimated Values');title('Missing Data Estimation');
err     = norm(estval - trueval)/length(estval);

%Check if the shared/unshared factors are identified accurately
data.adjlambda_rec

% Check how well extracted factors, i.e., Fac1 and Fac2, match with the original ones, data.Atrue. 
Fac1 = normalize(data.Zhat{1});
Fac2 = normalize(data.Zhat{2});
for i=1:3
    corr(Fac1{i},data.Atrue{i})    
end
corr(Fac2{2},data.Atrue{4})

%% References
% # E. Acar, T. G. Kolda, and D. M. Dunlavy, <http://arxiv.org/abs/1105.3422v1 All-at-once Optimization for
% Coupled Matrix and Tensor Factorizations>, KDD Workshop on Mining and Learning with Graphs, 2011 (arXiv:1105.3422v1).
% # E. Acar, A. J. Lawaetz, M. A. Rasmussen,and R. Bro, <http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6610925 Structure-Revealing Data Fusion Model with Applications in Metabolomics>, IEEE EMBC, pages 6023-6026, 2013.
% # D. M. Dunlavy, T. G. Kolda, and E. Acar, <https://software.sandia.gov/trac/poblano Poblano v1.0: A Matlab Toolbox for Gradient-Based Optimization>, SAND2010-1422, March 2010.
% # B. W. Bader, T. G. Kolda and others. <http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.5.html MATLAB Tensor Toolbox Version 2.5>, January 2012.
% # E. Acar, E. E. Papalexakis, G. Gurdeniz, M. Rasmussen, A. J. Lawaetz, M. Nilsson, and R. Bro, <http://www.biomedcentral.com/1471-2105/15/239 Structure-Revealing Data Fusion>, BMC Bioinformatics, 15: 239, 2014.        
