function y = Coupled_EEG_fMRI_STM_Predict(X_new, X_train, y_train, alpha, b, gammaU, gammaC, gammaV)
% Prediciton function for Coupled EEG_fMRI classification
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
       Ki(j) = Coupled_EEG_fMRI_Kernel_RBF(X_new{i}, X_train{j}, gammaU, gammaC, gammaV); 
   end
   yi = Ki * diag(y_train) * alpha + b;
   if yi >= 0
       y(i) = 1;
   else
       y(i) = -1;
   end
end

end