function [val] = Coupled_EEG_fMRI_Kernel_RBF(Tx1, Tx2, gammaU, gammaC, gammaV)
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

        T1_U = cell(modeU);
        T1_U{1} = Tx1.u1;
        T1_U{2} = Tx1.u2;
        
        T2_U = cell(modeU);
        T2_U{1} = Tx2.u1;
        T2_U{2} = Tx2.u2;
        
        T1_C = cell(modeC);
        T1_C{1} = Tx1.u3;
        T2_C = cell(modeC);
        T2_C{1} = Tx2.u3;
        
        T1_V = cell(modeV);
        T1_V{1} = Tx1.v1;
        
        T2_V = cell(modeV);
        T2_V{1} = Tx2.v1;
        
        val = 0.33 * TensorKernel_RBF(T1_U, T2_U, gammaU) + 0.5 * TensorKernel_RBF(T1_C, T2_C, gammaC) + 0.1 * TensorKernel_RBF(T1_V, T2_V, gammaV);
        
end