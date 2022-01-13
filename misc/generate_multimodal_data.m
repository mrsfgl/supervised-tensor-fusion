function [X, A] = generate_multimodal_data(sz, R, num_samples, modes, varargin)
% GENERATE_MULTIMODAL_DATA generates coupled or non-coupled multimodal data
% from factors generated from a Gaussian distribution.
%
%   [X, A] = create_coupled(sz, R, num_samples, modes) creates output data
%   where `sz` is a vector with row sizes of the generated factors. `R` is
%   the rank of both modalities. Thus a factor is a matrix of size sz(n)xR.
%   `num_samples` is the number of samples generated with the function, i.e.
%   length(X). `modes` indicates how the modes are coupled among each 
%   modality, e.g., {[1 2 3], [1 4], [2 5]} generates three modalities, 
%   where the third-order tensor (first modality) shares the first mode with
%   the first matrix (second modality) and the second mode with the second 
%   matrix (third modality).
%
%   Example
%   -------
%   sz = [10,20,30,40]
%   num_samples = 500
%   modes = {[1,2,3],[1,4]}  % This indicates that the first mode of both
%                              modalities are coupled.

if ~iscell(modes)
    modes = {modes};
end
M = length(modes);

params = inputParser;
params.addParameter('noise', 0, @isnumeric);
params.addParameter('missing_ratio', zeros(1, M), @isnumeric);
params.addParameter('rnd_seed', randi(10^3));

params.parse(varargin{:});
nlevel = params.Results.noise;
missing_ratio = params.Results.missing_ratio;
rnd_seed = params.Results.rnd_seed;

max_modeid = max(cellfun(@(x) max(x), modes));
if max_modeid ~= length(sz)
    error('Mismatch between size and modes inputs')
end

% Random seed
rng(rnd_seed);

%% Generate factor matrices
nb_modes  = length(sz);
% generate factor matrices
A = cell(num_samples, nb_modes);
for i = 1:num_samples
    for n = 1:nb_modes
        A{i}{n} = randn(sz(n), R);
        A{i}{n} = A{i}{n}*diag(sqrt(sum(A{i}{n}.^2)).^-1);
    end
end

%% Generate data blocks
X = cell(1,M);
for m = 1:M
    W = rand([num_samples,sz(modes{m})]) > missing_ratio(m);
    temp = zeros([num_samples,sz(modes{m})]);
    for i = 1:num_samples
        temp(i,:) = reshape(get_full_tensor(A{i}(modes{m})),[],1);
        temp(i,:) = temp(i,:) + nlevel * randn(1,prod(sz(modes{m})));
    end
    X{m}.orig = temp;
    X{m}.object = W.*temp;
    X{m}.miss = W;
    X{m}.modes = modes;
    X{m}.size = sz;
end


end