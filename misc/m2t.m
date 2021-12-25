function T = m2t(M, sz, n)
% T = m2t(M, sz, n)
% Matrix-to-Tensor reshaping operator.
mode_col = setdiff(1:length(sz),n);
T = reshape(M,[sz(n), sz(mode_col)]);
T = ipermute(T, [n, mode_col]);
end