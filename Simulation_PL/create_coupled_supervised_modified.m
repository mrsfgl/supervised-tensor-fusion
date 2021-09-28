function  [X, A] = create_coupled_supervised_modified(varargin)
% CREATE_COUPLED generates coupled higher-order tensors and matrices -
% and returns the generated data in two cells. The data are generated in
% a three mode tensor and a matrix coupled
%

%% Parse inputs
params = inputParser;
params.addParameter('size', [50 30 40 20], @isnumeric);
params.addParameter('modes', {[1 2 3], [1 4]}, @iscell);
params.addParameter('noise', 10, @isnumeric);
params.addParameter('n_samples', 30, @isnumeric);
params.addParameter('class_distance', 1, @isnumeric);
params.addParameter('class_var', .01, @isnumeric);
params.addParameter('flag_sparse',[0 0], @isnumeric);
params.addParameter('M',[0 0], @isnumeric);
params.addParameter('lambdas', {[1 1 1], [1 1 1]}, @iscell);
params.addParameter('rnd_seed', randi(10^3));
params.parse(varargin{:});
sz         = params.Results.size;    %size (shape) of data sets
lambdas    = params.Results.lambdas; % norms of components in each data set
modes      = params.Results.modes;   % how the data sets are coupled
nlevel     = params.Results.noise;
n_samples  = params.Results.n_samples;
flag_sparse = params.Results.flag_sparse;
M           = params.Results.M;
class_dist = params.Results.class_distance;
class_var  = params.Results.class_var;
rnd_seed   = params.Results.rnd_seed;


max_modeid = max(cellfun(@(x) max(x), modes));
if max_modeid ~= length(sz)
    error('Mismatch between size and modes inputs')
end

%% Random seed
rng(rnd_seed);

%% Generate factor matrices
nb_modes  = length(sz);
Rtotal    = length(lambdas{1});
A         = cell(2,1);
% generate factor matrices


for m = 1:n_samples
    for n = 1:nb_modes
        A{1}{m}{n} =  ones(sz(n), Rtotal) + randn(sz(n), Rtotal) + class_var*randn(sz(n), Rtotal);
        A{2}{m}{n} =  ones(sz(n), Rtotal) + randn(sz(n), Rtotal) + class_dist(n) + class_var*randn(sz(n), Rtotal);
        for r=1:Rtotal
            A{1}{m}{n}(:,r) = A{1}{m}{n}(:,r)/norm(A{1}{m}{n}(:,r));
            A{2}{m}{n}(:,r) = A{2}{m}{n}(:,r)/norm(A{2}{m}{n}(:,r));
        end
    end
end

%% Generate data blocks
P  = length(modes);
X  = cell(2,1);
for c = 1:2
    X{c} = cell(1,n_samples);
    for m = 1:n_samples
        for p = 1:P
            
            temp = full(ktensor(lambdas{p}',A{c}{m}(modes{p})));
            temp = awgn(double(temp), nlevel, 'measured');
            X{c}{m}{p} = temp;
        end
    end
end

