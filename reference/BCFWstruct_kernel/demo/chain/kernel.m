
function kv = kernel(param, x1f, x2f)

dif = (x1f-x2f);
norm_sq = sum(dif.*dif);

kv = exp(-param.gamma*norm_sq);

end %rbf-kernel
