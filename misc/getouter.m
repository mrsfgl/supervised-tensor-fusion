function out = getouter(bet, varargin)
% GETOUTER Takes the Kronecker product of vectors in cell array `bet`.
% Creates a vector `out` of size I1xI2x...xIN where N is the length of
% `bet`. If this vector is reshaped into a tensor with size I1, I2,..., IN,
% it becomes the rank-1 CP tensor of vectors in `bet`.

if isempty(varargin)
    j = NaN;
else
    j = varargin{1};
end
idx = setdiff(1:length(bet),j);
lens = cellfun(@numel,bet);
out = zeros(prod(lens(idx)),1);
len_vec = lens(idx(1));
out(1:len_vec) = bet{idx(1)};

idx = setdiff(idx,idx(1));
for k = 1:length(idx)
    next_len = lens(idx(k)) * len_vec;
    out(1:next_len) = kron(bet{idx(k)}, out(1:len_vec));
    len_vec = next_len;
end
end
