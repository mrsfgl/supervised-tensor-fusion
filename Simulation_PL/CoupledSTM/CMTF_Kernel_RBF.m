function [val] = CMTF_Kernel_RBF(Tx1, Tx2, gammaU, gammaC, gammaV)
%    Kernel function sepcially designed for Coupled EEG-fMRI data shaped
%    like our paper description. 
% Input: CP Tensor Components 
%         gammaU1: a vector for scaling parameter for component U from EEG
%         gammaC: a scalar scaling parameter for shared components 
%        gammaV: a scalar scaling parameter for component V from fMRI
% Ouput: a kernel value
        
        modeU = length(gammaU);
        modeC = length(gammaC);
        modeV = length(gammaV);

        T1_U = cell(1, modeU);
        T1_U{1} = Tx1.U{1};
        T1_U{2} = Tx1.U{2};
        
        T2_U = cell(1, modeU);
        T2_U{1} = Tx2.U{1};
        T2_U{2} = Tx2.U{2};
        
        T1_C = cell(1, modeC);
        T1_C{1} = Tx1.U{3};
        T2_C = cell(modeC);
        T2_C{1} = Tx2.U{3};
        
        T1_V = cell(1, modeV);
        T1_V{1} = Tx1.U{4};
        
        T2_V = cell(1, modeV);
        T2_V{1} = Tx2.U{4};
        
        val = 0.4 * TensorKernel_RBF(T1_U, T2_U, gammaU) + 0.2 * TensorKernel_RBF(T1_C, T2_C, gammaC) + 0.4 * TensorKernel_RBF(T1_V, T2_V, gammaV);
        
end