function y = coupled_stm_predict(X_new, X_train, y_train, alpha, b, gammaU, gammaC, gammaV)
% Prediction function for Coupled classification
% Input: new observations
%  Training set: X_train, y_train; alpha (column vector n * 1); b scalar
% Other parameters: tuning coefficients
% Output: predicted labels

n = length(X_new);
y = zeros(1, n);
m = length(X_train);
for i = 1 : n
   Ki = zeros(1, m);
   for j = 1 : m
       Ki(j) = coupled_kernel_rbf(X_new{i}, X_train{j}, gammaU, gammaC, gammaV); 
   end
   yi = Ki * diag(y_train) * alpha + b;
   if yi >= 0
       y(i) = 1;
   else
       y(i) = -1;
   end
end

end