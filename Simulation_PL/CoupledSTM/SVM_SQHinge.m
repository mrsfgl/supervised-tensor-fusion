function [alpha, b] = SVM_SQHinge(K, y, c, eta, maxitor)
% Squared Hinge Loss solving Support Vector Machine 
% Input: K: Kernel matrix, n by n
%        y: training labels n by 1 encoded as -1 and 1
%        c: parameter of slake variable (optional)
% Ouput: alpha: Lagrange Multiplier in a vector
%        b: offset


[n, ~] = size(K);
Dy = diag(y);

alpha_new = ones(n, 1);
current_eta = 10000;
current_itor = 1;
while current_eta > eta && current_itor < maxitor
    
    alpha_current = alpha_new;
    tmp = Dy * alpha_current;
    yc = K * tmp;
    sv_ind = double(yc < 1);
    Is = diag(sv_ind);
    inv_mm = pinv(c.* eye(n) + (1 / n).* Is * K);
    lf_mm = (1 / n).* Dy * inv_mm;
    rs_mm = Is * y';
    alpha_new = lf_mm * rs_mm;
    current_eta = norm(alpha_new - alpha_current);
    current_itor = current_itor + 1;
end
alpha = alpha_new;
for i = 1 : n
    if alpha(i) > 1.0e-5
        sv_idx = cat(1, sv_idx, i);
    else
        alpha(i) = 0;
    end
end
b = 0;
for b_idx = 1 : length(sv_idx)
    b = b + y(b_idx) - (alpha'.* y) * K(:, b_idx);
end
b = b / length(sv_idx);
end