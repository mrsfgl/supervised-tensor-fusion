function full_tens = get_full_tensor(U)
% GET_FULL_TENSOR creates full tensor X from factor matrices U.
%   
%   full_tens = get_full_tensor(U) 

if ~iscell(U) || ndims(U)<1
    error("Factor matrices have inapproptiate type or structure")
end
n_modes = length(U);
R = size(U{1},2);
sz = cellfun(@size, U, num2cell(ones(1,n_modes)));
full_tens = zeros(sz);
for r = 1:R
    temp = cell(1,n_modes);
    for n = 1:n_modes
        temp{n} = U{n}(:,r);
    end
    rank_r_vec = getouter(temp);
    full_tens = full_tens + reshape(rank_r_vec, sz);
end
        