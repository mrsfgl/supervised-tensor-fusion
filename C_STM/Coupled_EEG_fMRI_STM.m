function [alpha, b] = Coupled_EEG_fMRI_STM(X, y, C, gammaU, gammaC, gammaV)
% Fit hinge loss based STM for model

    n = length(X);
    K = zeros(n, n);
    for i = 1 : n
       for j = i : n
          K(i, j) = Coupled_EEG_fMRI_Kernel(X{i}, X{j}, gammaU, gammaC, gammaV);
          K(j, i) = K(i, j);
       end
    end
    [alpha, b] = SVM_slover(K, y, C);

end