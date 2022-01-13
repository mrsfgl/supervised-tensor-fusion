function out = eval_reg(B, im)
% out = eval_reg(B, im)
% Evaluates regression performance by comparing the estimate B to original
% regressor im.

sizes = size(B);
% min_B = min(B, [], 'all');
% max_B = max(B, [], 'all');
% B = (B-min_B)./(max_B-min_B);
out.rmse_B = sqrt(sum((B-im).^2,'all')/prod(sizes));
out.norm_err_B = sum((B-im).^2,'all')/norm(im(:)).^2;
out.norm_1_B = sum(abs(B-im),'all')/norm(im(:),1);
end