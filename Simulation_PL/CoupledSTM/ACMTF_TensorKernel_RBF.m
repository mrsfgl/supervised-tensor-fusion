function val = ACMTF_TensorKernel_RBF(Tx1, Tx2, gamma, varargin)
% CP-Tensor Gaussian RBF Kernel using definition in Li & Maiti 2019
% Input: Tensor CP Kernel in cell (1 x d), where d is the mode
%         gamma: scaling parameter for each mode of the tensor
% Output: Tensor kernel value
        tmp = Tx1{1};
        [~, T_rank] = size(tmp); 
        T_mode = length(Tx1);
        
        val = 0;
        for i = 1 : T_rank
           for j = 1 : T_rank
              tmp_val = 1;
              for k = 1 : T_mode
                 compX = Tx1{k};
                 compY = Tx2{k};
                 vec1 = compX(:, i);
                 vec2 = compY(:, j);
                 tmp_val = tmp_val * exp(-1 * gamma(k) * norm(vec1 - vec2)^2);
              end
              val = val + tmp_val;
           end
        end
end