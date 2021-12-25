function mu_B = getmean(X, beta, rank, p, varargin)

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