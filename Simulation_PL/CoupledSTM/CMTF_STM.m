function [alpha, b] = CMTF_STM(X, y, C, gammaU, gammaC, gammaV, solver)
% Fit hinge loss based STM for model

    n = length(X);
    K = zeros(n, n);
    for i = 1 : n
       for j = i : n
          K(i, j) = CMTF_Kernel_RBF(X{i}, X{j}, gammaU, gammaC, gammaV);
          K(j, i) = K(i, j);
       end
    end
    if solver == 'QP'
        [alpha, b] = SVM_slover(K, y, C);
    else
        [alpha, b] = SVM_SQHinge(K, y, C);
    end

end