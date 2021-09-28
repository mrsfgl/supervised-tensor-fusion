function [err, corrs] = err_analysis_acmtf(data, modes, flag_soft)


P = length(data.Xorig);
Facs = cell(1,length(unique(cell2mat(modes))));
Zhat = data.Zhat;
fms = zeros(1,P);
for p=1:P
    for i=1:length(modes{p})
        Facs{modes{p}(i)} = data.Zhat{p}{i};
    end
    Zhat{p} = full(Zhat{p});
    fms(p) = score(ktensor(Facs(modes{p})), ktensor(data.Atrue(modes{p})));
end
TrueFacs = data.Atrue;
modes = cell2mat(modes);

corrs = zeros(1,length(TrueFacs));
if flag_soft
    for n=1:length(TrueFacs)
        Cmat = corr(Facs{modes(n)},TrueFacs{n});
        corrs(n) = mean(max(abs(Cmat)));
    end
else
    for n=1:length(TrueFacs)
        Cmat = corr(Facs{n},TrueFacs{n});
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