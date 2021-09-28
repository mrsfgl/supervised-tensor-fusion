function [alpha, b] = coupled_stm(X, y, C, gammaU, gammaC, gammaV)
% Fit hinge loss based STM for model

    n = length(X);
    K = zeros(n, n);
    for i = 1 : n
       for j = i : n
          K(i, j) = coupled_kernel_rbf(X{i}, X{j}, gammaU, gammaC, gammaV);
          K(j, i) = K(i, j);
       end
    end
    [alpha, b] = SVM_slover(K, y, C);

end