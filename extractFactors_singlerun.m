function [F, output] = extractFactors_singlerun(imsMRI, imfMRI, CP_rank)
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

model.factorizations.tensor2.data = imfMRI;
model.factorizations.tensor2.cpd = {'V1','V2','V3','V4'};

% Optimize
options.Display = 50;
options.TolX = 10^-4;
options.TolFun = 10^-4;
options.MaxIter = 100;
[F, output] =  sdf_minf(model, options);
end