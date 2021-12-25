function Y = downsample_mean(X, sample_rate)

sz = size(X);
end_sz = sz;
N = length(sz);
Y = X;

for n=1:N
    end_sz(n) = ceil(sz(n)/sample_rate);
    rem = end_sz(n)*sample_rate-sz(n);
    av_mat = zeros(sz(n),end_sz(n));
    for j=1:end_sz(n)-1
        av_mat((j-1)*sample_rate+1:j*sample_rate, j) = 1/sample_rate;
    end
    if rem == 0
        av_mat(sz(n)-sample_rate+1:sz(n), end_sz(n)) = 1/sample_rate;
    else
        av_mat(j*sample_rate+1:sz(n), end_sz(n)) = 1/rem;
    end
    Y = m2t(av_mat'*t2m(Y, n), end_sz, n);
end

end