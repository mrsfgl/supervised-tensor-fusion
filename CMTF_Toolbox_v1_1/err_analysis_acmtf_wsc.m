function [err, corrs] = err_analysis_acmtf_wsc(data, modes, flag_soft)


P = length(data.Xorig);
Facs = cell(1,length(unique(cell2mat(modes))));
Zhat = data.Zhat;
ind_last = 0;
for p=1:P
    for i=1:length(modes{p})
        Facs{i+ind_last} = data.Zhat{p}{i};
    end
    ind_last = ind_last + length(modes{p});
    Zhat{p} = full(Zhat{p});
end
modes = cell2mat(modes);
TrueFacs = data.Atrue;

corrs = zeros(1,length(TrueFacs));
if flag_soft
    for n=1:length(TrueFacs)
        Cmat = corr(Facs{n},TrueFacs{n});
        corrs(n) = mean(max(abs(Cmat)));
    end
else
    for n=1:length(TrueFacs)
        Cmat = corr(Facs{n},TrueFacs{modes(n)});
        corrs(n) = mean(max(abs(Cmat)));
    end
end

err = ones(1,P);
for p = 1:P
    trueval = data.Xorig{p}(find(data.W{p}==0));
    estval = Zhat{p}(find(data.W{p}==0));
    % plot(trueval,estval,'*');xlabel('True Values');ylabel('Estimated Values'); title('Missing Data Estimation');
    err(p) = norm(estval - trueval)/length(estval);
end
end