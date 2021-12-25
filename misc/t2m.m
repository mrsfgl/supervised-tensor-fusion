function M = t2m(T, m)
% M = t2m(T, m)
% Folds a tensor T to a matrix M at given modes m where m corresponds to
% modes that are going to be stacked in the row of M.
% T : Input tensor.
% m : modes that will be stacked in the row.
s = size(T);
if length(s)<max(m)
    s(end+1:max(m)) = 1;
end
m_not = setdiff(1:length(s), m);
T = permute(T, [m, m_not]);
M = reshape(T, prod(s(m)), []);
end