function plot_noise_exps(results, noise_levels)

n_exps = size(results,1);
n_noise = length(noise_levels);
K = length(results{1});

accs = zeros(5, n_noise, K, n_exps);
precs = zeros(5, n_noise, K, n_exps);
recalls = zeros(5, n_noise, K, n_exps);
specs = zeros(5, n_noise, K, n_exps);
errs = zeros(5, n_noise, K, n_exps);
for i=1:n_exps
    for j = 1:n_noise
        for k = 1:K
            accs(1,j,k,i) = results{i,1,j,1}(k).cp.accuracy;
            accs(2,j,k,i) = results{i,1,j,1}(k).pca.accuracy;
            accs(3,j,k,i) = results{i,1,j,1}(k).cmtf.accuracy;
            accs(4,j,k,i) = results{i,1,j,1}(k).acmtf.accuracy;
            accs(5,j,k,i) = results{i,1,j,1}(k).acmtf_sc.accuracy;
            
            precs(1,j,k,i) = results{i,1,j,1}(k).cp.precision;
            precs(2,j,k,i) = results{i,1,j,1}(k).pca.precision;
            precs(3,j,k,i) = results{i,1,j,1}(k).cmtf.precision;
            precs(4,j,k,i) = results{i,1,j,1}(k).acmtf.precision;
            precs(5,j,k,i) = results{i,1,j,1}(k).acmtf_sc.precision;
            
            recalls(1,j,k,i) = results{i,1,j,1}(k).cp.recall;
            recalls(2,j,k,i) = results{i,1,j,1}(k).pca.recall;
            recalls(3,j,k,i) = results{i,1,j,1}(k).cmtf.recall;
            recalls(4,j,k,i) = results{i,1,j,1}(k).acmtf.recall;
            recalls(5,j,k,i) = results{i,1,j,1}(k).acmtf_sc.recall;
            
            specs(1,j,k,i) = results{i,1,j,1}(k).cp.specificity;
            specs(2,j,k,i) = results{i,1,j,1}(k).pca.specificity;
            specs(3,j,k,i) = results{i,1,j,1}(k).cmtf.specificity;
            specs(4,j,k,i) = results{i,1,j,1}(k).acmtf.specificity;
            specs(5,j,k,i) = results{i,1,j,1}(k).acmtf_sc.specificity;
            
            errs(1,j,k,i) = mean(results{i,1,j,1}(k).cp.err);
            errs(2,j,k,i) = mean(results{i,1,j,1}(k).pca.err);
            errs(3,j,k,i) = mean(results{i,1,j,1}(k).cmtf.err(:,1));
            errs(4,j,k,i) = mean(results{i,1,j,1}(k).acmtf.err(:,1));
            errs(5,j,k,i) = mean(results{i,1,j,1}(k).acmtf_sc.err(:,1));
        end
    end
end
precs(isnan(precs)) = 0;

m_accs = mean(mean(accs,3), 4); s_accs = std(mean(accs,3), [], 4);
m_precs = mean(mean(precs,3), 4); s_precs = std(mean(precs,3), [], 4);
m_recalls = mean(mean(recalls,3), 4); s_recalls = std(mean(recalls,3), [], 4);
m_specs = mean(mean(specs,3), 4); s_specs = std(mean(specs,3), [], 4);
m_errs = mean(mean(errs,3), 4); s_errs = std(mean(errs,3), [], 4);

figure
ax = subplot(2,2,1);
b = bar(noise_levels, m_accs');
hold
for i=1:length(b)
    for k=1:size(m_accs,2)
        errorbar(b(i).XData(k)+b(i).XOffset, m_accs(i,k), s_accs(i,k),'k')
    end
end
% errorbar(noise_levels, m_accs(1,:), s_accs(1,:), '-*', 'LineWidth', 2, 'DisplayName', 'CP')
% hold
% errorbar(noise_levels, m_accs(2,:), s_accs(2,:), '--', 'LineWidth', 2, 'DisplayName', 'PCA')
% errorbar(noise_levels, m_accs(3,:), s_accs(3,:), '-x', 'LineWidth', 2, 'DisplayName', 'CMTF')
% errorbar(noise_levels, m_accs(4,:), s_accs(4,:), '--x', 'LineWidth', 2, 'DisplayName', 'ACMTF')
% errorbar(noise_levels, m_accs(5,:), s_accs(5,:), '--*', 'LineWidth', 2, 'DisplayName', 'ACMTF_sc')
ax.FontSize = 17;
xlabel('Noise Levels (SNR)')
ylabel('Accuracy')
legend('CP', 'PCA', 'CMTF', 'ACMTF', 'ACMTF-SC', 'location', 'southeast')

ax= subplot(2,2,2);
b = bar(noise_levels, m_precs');
hold
for i=1:length(b)
    for k=1:size(m_accs,2)
        errorbar(b(i).XData(k)+b(i).XOffset, m_precs(i,k), s_precs(i,k),'k')
    end
end
% errorbar(noise_levels, m_precs(1,:), s_precs(1,:), '-*', 'LineWidth', 2, 'DisplayName', 'CP')
% hold
% errorbar(noise_levels, m_precs(2,:), s_precs(2,:), '--', 'LineWidth', 2, 'DisplayName', 'PCA')
% errorbar(noise_levels, m_precs(3,:), s_precs(3,:), '-x', 'LineWidth', 2, 'DisplayName', 'CMTF')
% errorbar(noise_levels, m_precs(4,:), s_precs(4,:), '--x', 'LineWidth', 2, 'DisplayName', 'ACMTF')
% errorbar(noise_levels, m_precs(5,:), s_precs(5,:), '--*', 'LineWidth', 2, 'DisplayName', 'ACMTF_sc')
% grid,
ax.FontSize = 17;
xlabel('Noise Levels (SNR)')
ylabel('Precision')
legend('CP', 'PCA', 'CMTF', 'ACMTF', 'ACMTF-SC', 'location', 'southeast')

ax = subplot(2,2,3);
b = bar(noise_levels, m_recalls');
hold
for i=1:length(b)
    for k=1:size(m_accs,2)
        errorbar(b(i).XData(k)+b(i).XOffset, m_recalls(i,k), s_recalls(i,k),'k')
    end
end
% errorbar(noise_levels, m_recalls(1,:), s_recalls(1,:), '-*', 'LineWidth', 2, 'DisplayName', 'CP')
% hold
% errorbar(noise_levels, m_recalls(2,:), s_recalls(2,:), '--', 'LineWidth', 2, 'DisplayName', 'PCA')
% errorbar(noise_levels, m_recalls(3,:), s_recalls(3,:), '-x', 'LineWidth', 2, 'DisplayName', 'CMTF')
% errorbar(noise_levels, m_recalls(4,:), s_recalls(4,:), '--x', 'LineWidth', 2, 'DisplayName', 'ACMTF')
% errorbar(noise_levels, m_recalls(5,:), s_recalls(5,:), '--*', 'LineWidth', 2, 'DisplayName', 'ACMTF_sc')
% grid,
ax.FontSize = 17;
xlabel('Noise Levels (SNR)')
ylabel('Recall(Sensitivity)')
legend('CP', 'PCA', 'CMTF', 'ACMTF', 'ACMTF-SC', 'location', 'southeast')

ax = subplot(2,2,4);
b = bar(noise_levels, m_specs');
hold
for i=1:length(b)
    for k=1:size(m_accs,2)
        errorbar(b(i).XData(k)+b(i).XOffset, m_specs(i,k), s_specs(i,k),'k')
    end
end
% errorbar(noise_levels, m_specs(1,:), s_specs(1,:), '-*', 'LineWidth', 2, 'DisplayName', 'CP')
% hold
% errorbar(noise_levels, m_specs(2,:), s_specs(2,:), '--', 'LineWidth', 2, 'DisplayName', 'PCA')
% errorbar(noise_levels, m_specs(3,:), s_specs(3,:), '-x', 'LineWidth', 2, 'DisplayName', 'CMTF')
% errorbar(noise_levels, m_specs(4,:), s_specs(4,:), '--x', 'LineWidth', 2, 'DisplayName', 'ACMTF')
% errorbar(noise_levels, m_specs(5,:), s_specs(5,:), '--*', 'LineWidth', 2, 'DisplayName', 'ACMTF_sc')
% grid,
ax.FontSize = 17;
xlabel('Noise Levels (SNR)')
ylabel('Specificity')
legend('CP', 'PCA', 'CMTF', 'ACMTF', 'ACMTF-SC', 'location', 'southeast')

figure;
b = bar(noise_levels, m_errs');
hold
for i=1:length(b)
    for k=1:size(m_accs,2)
        errorbar(b(i).XData(k)+b(i).XOffset, m_errs(i,k), s_errs(i,k),'k')
    end
end
% errorbar(noise_levels, m_specs(1,:), s_specs(1,:), '-*', 'LineWidth', 2, 'DisplayName', 'CP')
% hold
% errorbar(noise_levels, m_specs(2,:), s_specs(2,:), '--', 'LineWidth', 2, 'DisplayName', 'PCA')
% errorbar(noise_levels, m_specs(3,:), s_specs(3,:), '-x', 'LineWidth', 2, 'DisplayName', 'CMTF')
% errorbar(noise_levels, m_specs(4,:), s_specs(4,:), '--x', 'LineWidth', 2, 'DisplayName', 'ACMTF')
% errorbar(noise_levels, m_specs(5,:), s_specs(5,:), '--*', 'LineWidth', 2, 'DisplayName', 'ACMTF_sc')
% grid,
ax = gca;
ax.FontSize = 17;
xlabel('Noise Levels (SNR)')
ylabel('Residual Error')
legend('CP', 'PCA', 'CMTF', 'ACMTF', 'ACMTF-SC', 'location', 'southeast')
end