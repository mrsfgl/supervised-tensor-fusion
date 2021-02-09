function [a, b] = KMeans_Clustering_Compare(imMRI, imfMRI, activity, CP_rank)
% Perform both coupled tensor decomposition & clustering and regular CP
% decompostion & clustering using K-means clustering methods; Return Rand
% index of two different pipeline for comparison

num_clusters = 3;

[F, ~] = extractFactors_singlerun(imMRI, imfMRI, CP_rank);
coupled_feature = F.factors.V4;

fMRI_dim = size(imfMRI);
total_sti = fMRI_dim(4);
sti_label = zeros(length(activity), 1);


for i = 1 : length(sti_label)
   if activity(i, :) == 'standard'
       sti_label(i) = 1;
   elseif activity(i, :) == 'target  '
       sti_label(i) = 2;
   else
       sti_label(i) = 3;
   end
end

coupled_feature = coupled_feature(total_sti - length(sti_label) + 1 : end, :);
coupled_pred = kmeans(coupled_feature.', num_clusters);
a = rand_index(sti_label, coupled_pred);

Vnew = cpd(imfMRI, CP_rank);
feature_CP =  Vnew{4};
feature_CP = feature_CP(total_sti - length(sti_label) + 1 : end, :);
pred_CP = kmeans(feature_CP.', num_clusters);
b = rand_index(sti_label, pred_CP);
end