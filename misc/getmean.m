function mu_B = getmean(X, beta, rank, p, varargin)
% mu_B = getmean(X, beta, rank, p)
% Computes the full tensor B from its rank 1 constituents beta
% and projects the X to this full tensor.

if isempty(varargin)
    rank_exclude = nan;
else
    rank_exclude = varargin{1};
end

idx = setdiff(1:rank,rank_exclude);
B = getouter(beta(idx(1),:));
for i = 2:length(idx)
    B = B + getouter(beta(idx(i),:));
end
B = reshape(B, p);
mu_B = t2m(X,1)*B(:);
end