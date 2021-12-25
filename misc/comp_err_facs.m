function [err, corrs, fms] = comp_err_facs(X, F, Y, modes, varargin)

if length(X)~=length(F) || ~iscell(X) || ~iscell(F)
    error('Wrong input type!')
end
N = length(X);
corrs = zeros(1,N);
for n=1:N
    C = corr(X{n}, F{n});
    corrs(n) = mean(max(abs(C)));
end

P = length(modes);
fms = zeros(1,length(unique(cell2mat(modes))));
for p=1:P
    for n=1:length(modes{p})
        fms(modes{p}(n)) = score(ktensor(X(modes{p}(n))), ktensor(F(modes{p}(n))));
    end
end

err = zeros(1,length(modes));
if nargin==4
    for p=1:P
        found = ktensor(F(modes{p}));
        err(p) = norm(Y{p}-full(found))/norm(Y{p});
    end
else
    T = varargin{1};
    for p=1:P
        found = full(T{p});
        err(p) = norm(Y{p}-full(found))/norm(Y{p});
    end
end
end
