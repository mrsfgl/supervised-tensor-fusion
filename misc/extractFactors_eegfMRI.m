function [F, output] = extractFactors_eegfMRI(eeg, fMRI, CP_rank, lambda1, lambda2)
% [F, output] = extractFactors_eegfMRI(eeg, fMRI)
% Extracts coupled and uncoupled factors of structural and functional MRI 
% For a single run 
R = CP_rank;
sz_s = size(eeg);
sz_f = size(fMRI);

% Define model variables
model = struct;
model.variables.u1 = randn(sz_s(1),R);
model.variables.u2 = randn(sz_s(2),R);
model.variables.u3 = randn(sz_s(3),R);
model.variables.v1 = randn(sz_f(1),R);


% Define subsampling matrices

% Define factor relations to variables.
model.factors.U1 = {'u1'};
model.factors.U2 = {'u2'};
model.factors.U3 = {'u3'};
model.factors.V1 = {'v1'};
model.factors.V2 = {'u3'};

% Define tensors and their factors to be learned.
model.factorizations.tensor1.data = eeg;
model.factorizations.tensor1.cpd = {'U1','U2','U3'};
model.factorizations.tensor1.weight = lambda1;

model.factorizations.tensor2.data = fMRI;
model.factorizations.tensor2.cpd = {'V1','V2'};
model.factorizations.tensor2.weight = lambda2;

% Optimize
options.Display = 100;
options.TolX = 10^-6;
options.TolFun = 10^-6;
options.MaxIter = 10^3;
[F, output] =  sdf_minf(model, options);
end