function [alpha, b] = SVM_slover(K, y, c)
% Quadratic programming solving Support Vector Machine 
% Input: K: Kernel matrix, n by n
%        y: training labels n by 1 encoded as -1 and 1
%        c: parameter of slake variable (optional)
% Ouput: alpha: Lagrange Multiplier in a vector
%        b: offset

if nargin < 3
    c = 0;
end
[n, ~] = size(K);
Dy = diag(y);
H = Dy * K * Dy;
f = ones(n, 1).* -1;
A = eye(n).* -1;
b = zeros(n, 1);
Aeq = y;
beq = 0;
if c > 0
   ub = ones(n, 1).* c;
else
    ub = [];
end
    
alpha = quadprog(H, f, A, b, Aeq, beq, [], ub);
sv_idx = [];
for i = 1 : n
    if alpha(i) > 1.0e-5
        sv_idx = cat(1, sv_idx, i);
    else
        alpha(i) = 0;
    end
end
b_idx = randsample(sv_idx, 1);
b = y(b_idx) - (alpha.* y)' * K(:, b_idx);
end