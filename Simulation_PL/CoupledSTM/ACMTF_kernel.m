function val = ACMTF_kernel(Tx1, Tx2, gammaU, gammaC, gammaV)
  %    Kernel function sepcially designed for Coupled EEG-fMRI data
  %    decomposed by ACMTF function
% Input: CP Tensor Components 
%         gammaU1: a vector for scaling parameter for component U from EEG
%         gammaC: a scalar scaling parameter for shared components 
%        gammaV: a scalar scaling parameter for component V from fMRI
% Ouput: a kernel value
        
        T1_U = cell(1, 2);
        T1_U{1} = Tx1{1}.U{1};
        T1_U{2} = Tx1{1}.U{2};
        
        T2_U = cell(1, 2);
        T2_U{1} = Tx2{1}.U{1};
        T2_U{2} = Tx2{1}.U{2};
        
        T1_C = cell(1);
        T1_C{1} = (Tx1{1}.U{3} + Tx1{2}.U{2})./ 2;
        T2_C = cell(1);
        T2_C{1} = (Tx2{1}.U{3} + Tx2{2}.U{2})./ 2;
        
        T1_V = cell(1);
        T1_V{1} = Tx1{2}.U{1};
        
        T2_V = cell(1);
        T2_V{1} = Tx2{2}.U{1};

    val = 0.4 * TensorKernel_RBF(T1_U, T2_U, gammaU) + 0.2 * TensorKernel_RBF(T1_C, T2_C, gammaC) + 0.4  * TensorKernel_RBF(T1_V, T2_V, gammaV);

end