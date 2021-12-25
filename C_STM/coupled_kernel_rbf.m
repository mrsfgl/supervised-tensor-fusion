function [val] = coupled_kernel_rbf(Tx1, Tx2, gammaU, gammaC, gammaV)
%    Kernel function designed for Coupled data 
% Input: CP Tensor Components 
%        gammaU: a vector for scaling parameter for component U from first
%        mode
%        gammaC: a scalar scaling parameter for shared components 
%        gammaV: a scalar scaling parameter for component V from the second
%        mode
% Ouput: a kernel value
        
if isfield(Tx1, 'u1')
    T1_U{1} = Tx1.u1; T1_U{2} = Tx1.u2; T2_U{1} = Tx2.u1; T2_U{2} = Tx2.u2;
    T1_C{1} = Tx1.u3; T2_C{1} = Tx2.u3; T1_V{1} = Tx1.v1; T2_V{1} = Tx2.v1;

    val = gammaU * TensorKernel_RBF(T1_U, T2_U, 1) +...
        gammaC * TensorKernel_RBF(T1_C, T2_C, 1) + ...
        gammaV * TensorKernel_RBF(T1_V, T2_V, 1);
    % gamma = 1 / (dim * var(factor))
else
    if iscell(Tx1)
        if length(Tx1)==2 && any(matches(class(Tx1{1}),'ktensor','IgnoreCase',true))
    %         U1 = Tx1{1}.U; U1(end+1) = Tx1{2}.U(1);
    %         U2 = Tx2{1}.U; U2(end+1) = Tx2{2}.U(1);
    %         val = TensorKernel_RBF(U1, U2, [gammaU,gammaC,gammaV]);
            val = gammaC*TensorKernel_RBF(Tx1{1}.U(3), Tx2{1}.U(3), 1) +...
                sum(gammaU)*TensorKernel_RBF(Tx1{1}.U(1:2), Tx2{1}.U(1:2), 1) +...
                sum(gammaV)*TensorKernel_RBF(Tx1{2}.U(1:end-1), Tx2{2}.U(1:end-1), 1);
        else
            val = TensorKernel_RBF(Tx1, Tx2, [gammaU,gammaC]);
        end
    else
        if length(Tx1.U)==4
%             val = TensorKernel_RBF(Tx1.U, Tx2.U, [gammaU,gammaC,gammaV]);
            val = sum(gammaU)*TensorKernel_RBF(Tx1.U(1:2), Tx2.U(1:2), 1) +...
                gammaC*TensorKernel_RBF(Tx1.U(3), Tx2.U(3), 1) +...
                sum(gammaV)*TensorKernel_RBF(Tx1.U(4), Tx2.U(4), 1);
        elseif length(Tx1.U)==5
            val = gammaU*TensorKernel_RBF(Tx1.U(1:2), Tx2.U(1:2),1) +...
                gammaC*TensorKernel_RBF(Tx1.U([3,5]), Tx2.U([3,5]),1)+...
                gammaV*TensorKernel_RBF(Tx1.U(4), Tx2.U(4),1);
        else
            val = sum(gammaU)*TensorKernel_RBF(Tx1.U(1:2), Tx2.U(1:2), 1)+...
                gammaC*TensorKernel_RBF(Tx1.U(3), Tx2.U(3), 1);
        end
    end
end
end