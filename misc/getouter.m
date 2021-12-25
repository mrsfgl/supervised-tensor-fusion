function out = getouter(bet, varargin)
if isempty(varargin)
    j = NaN;
else
    j = varargin{1};
end
idx = setdiff(1:length(bet),j);
out = bet{idx(1)};

idx = setdiff(idx,idx(1));
for k = 1:length(idx)
    out = kron(bet{idx(k)}, out);
end
end
