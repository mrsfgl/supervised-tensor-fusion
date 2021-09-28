function [val] = TensorKernel_RBF(Tx1, Tx2, gamma, varargin)
% CP-Tensor Gaussian RBF Kernel using definition in Li & Maiti 2019
% Input: Tensor CP Kernel in cell (1 x d), where d is the mode
%         gamma: scaling parameter for each mode of the tensor
% Output: Tensor kernel value
tmp = Tx1{1};
[~, T_rank] = size(tmp); 
T_mode = length(Tx1);

% 
% for k = 1 : T_mode
%  const = floor(log10(sum([var(Tx1{k}),var(Tx2{k})])));
%  gamma(k) = 1/(size(Tx1{k},1)*10^const);
% end
% val = zeros(T_rank);
% for i = 1 : T_rank
%    for j = 1 : T_rank
%       tmp_val = 1;
%       for k = 1 : T_mode
%          vec1 = Tx1{k}(:, i);
%          vec2 = Tx2{k}(:, j);
%          tmp_val = tmp_val * exp(-1 * gamma(k) * norm(vec1 - vec2)^2);
%       end
%       val(i,j) = tmp_val;
%    end
% end
% val = sum(val(:));

T1 = ktensor(Tx1);
T2 = ktensor(Tx2);

val = score(T1,T2);
end