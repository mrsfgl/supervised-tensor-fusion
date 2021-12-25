function [c,time] = efficient_corcondia(X,Fac,sparse_flag)
%Vagelis Papalexakis - Carnegie Mellon University, School of Computer
%Science (2014)
%This is an efficient algorithm for computing the CORCONDIA diagnostic for
%the PARAFAC decomposition (Bro and Kiers, "A new
%efficient method for determining the number of components in PARAFAC
%models", Journal of Chemometrics, 2003)
%This algorithm is part of a paper to be submitted to IEEE ICASSP 2015
if nargin == 2
    sparse_flag = 0;
end

F = size(Fac.U{1},2);
sz = size(X);

tic
for n=1:ndims(X)
    if n==1
        M = Fac.U{n}*diag(Fac.lambda);
    else
        M = Fac.U{n};
    end
    if(sparse_flag)
        [U{n} S{n} V{n}] = svds(M, F);
    else
        [U{n} S{n} V{n}] = svd(M,'econ');
    end
end

part1 = kron_mat_vec(cellfun(@transpose, U, 'UniformOutput', false), X);
part2 = kron_mat_vec(cellfun(@pinv, S, 'UniformOutput', false), part1);
G = kron_mat_vec(V, part2);

T = sptensor(F*ones(1,ndims(X)));
subs = cell(1,ndims(X));
for i = 1:F 
    ind = cumprod([1,sz(1:end-1)])*((i-1)*ones(ndims(X),1))+1;
    [subs{:}] = ind2sub(sz, ind);
    T(subs{:}) = 1; 
end

c = 100* (1 - sum((double(G-T).^2), 'all')/F);
time = toc;
end

function C = kron_mat_vec(Alist,X)
K = length(Alist);
for k = K:-1:1
    A = Alist{k};
    Y = ttm(X,A,k);
    X = Y;
end
C = Y;
end