function [err, corrs] = err_analysis_cmtf(data, modes, flag_soft)


P = length(data.Xorig);
Facs = data.Fac;
TrueFacs = data.Factrue;

Zhat = cell(P,1);
for p = 1:P
    Zhat{p} = full(ktensor(Facs(modes{p})));
end

modes = cell2mat(modes);
if flag_soft
    corrs = zeros(1,length(TrueFacs));
    for n=1:length(TrueFacs)
        Cmat = corr(Facs{modes(n)},TrueFacs{n});
        corrs(n) = mean(max(abs(Cmat)));
    end
else
    corrs = zeros(1,length(TrueFacs));
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