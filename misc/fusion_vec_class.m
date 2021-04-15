
clear fMRI_stacked

l = size(oddball{2,1},4);
sz = size(oddball{2,1});
fMRI_stacked(:,:,:,1:l,:) = cat(5,oddball{2,1}, standard{2,1});
fMRI_stacked(:,:,:,l+1:2*l,:) = cat(5,oddball{2,2}, standard{2,2});
fMRI_stacked(:,:,:,2*l+1:3*l,:) = cat(5,oddball{2,3}, standard{2,3});


fMRI_stacked = reshape(fMRI_stacked, [sz(1:3),l*6]);
labels = [ones(l*3,1); -ones(l*3,1)];

ranks = [3,5,7,9,11];
w = [10.^[-1:-1:-4], 0];
for i=1:5
    for j=1:5
        [F, output] = extractFactors_singlerun(imsMRI, fMRI_stacked, ranks(i), w(j), 1);
        coupled_features = F.factors.V4;
        
        SVM_Model = fitcsvm(coupled_features, labels);
        CVSVMModel = crossval(SVM_Model);
        classLoss(i,j) = kfoldLoss(CVSVMModel);
    end
end


function [F, output] = extractFactors_singlerun(imsMRI, imfMRI, CP_rank, lambda1, lambda2)
% [F, output] = extractFactors(imsMRI, imfMRI)
% Extracts coupled and uncoupled factors of structural and functional MRI 
% For a single run 
R = CP_rank;
sz_s = size(imsMRI);
sz_f = size(imfMRI);

% Define model variables
model = struct;
model.variables.u1 = randn(sz_s(1),R);
model.variables.u2 = randn(sz_s(2),R);
model.variables.u3 = randn(sz_s(3),R);
model.variables.v1 = randn(sz_f(4),R);


% Define subsampling matrices
r_horizontal = ceil(sz_s(1)/sz_f(1));
avM = kron(eye(sz_f(1)), ones(r_horizontal,1))/r_horizontal;
r_vertical = ceil(sz_s(3)/sz_f(3));
avH = kron(eye(sz_f(3)), ones(r_vertical,1))/r_vertical;
avH = avH(1:sz_s(3),:);

% Define factor relations to variables.
model.factors.U1 = {'u1'};
model.factors.U2 = {'u2'};
model.factors.U3 = 'u3';
model.factors.V1 = {'u1', @(z, task) struct_matvec(z,task,avM')};
model.factors.V2 = {'u2', @(z, task) struct_matvec(z,task,avM')};
model.factors.V3 = {'u3', @(z, task) struct_matvec(z,task,avH')};
model.factors.V4 = 'v1';

% Define tensors and their factors to be learned.
model.factorizations.tensor1.data = imsMRI;
model.factorizations.tensor1.cpd = {'U1','U2','U3'};
model.factorizations.tensor1.weight = lambda1;

model.factorizations.tensor2.data = imfMRI;
model.factorizations.tensor2.cpd = {'V1','V2','V3','V4'};
model.factorizations.tensor2.weight = lambda2;

% Optimize
options.Display = 50;
options.TolX = 10^-5;
options.TolFun = 10^-6;
options.MaxIter = 10^3;
[F, output] =  sdf_minf(model, options);
end