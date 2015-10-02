function product = kernel_inner_prod( param, model, x_f, y_f, ins_id, factor_id )

node_id = model.offset(ins_id) + factor_id;

product = 0.0;
alphas_y_f = model.alpha(:,y_f+1);

if( nnz(alphas_y_f) == 0 )
	return;
end

for i = find(alphas_y_f)'
	kv = model.kernel_cache(i,node_id);
	if( kv == 0 )
		kv = kernel( param, model.patterns{i}, x_f );
		model.kernel_cache(i,node_id) = kv;
	end
	product = product + alphas_y_f(i) * kv;
end

%{
for i = 1:model.n
	for j = 1:model.L(i)
		
		if model.alpha{i}{j}(y_f+1) == 0
			continue;
		end
		
		kv = model.kernel_cache{ins_id}{factor_id}(y_f+1);
		if( kv == 0 )
			kv = kernel( param, model.patterns{i}.data(:,j), x_f );
			model.kernel_cache{ins_id}{factor_id}(i,j) = kv;
		end
		product = product + model.alpha{i}{j}(y_f+1) * kv;
	end
end
%}

end %inner_prod

function kv = kernel(param, x1f, x2f)

dif = (x1f-x2f);
norm_sq = sum(dif.*dif);
kv = exp(-param.gamma*norm_sq);

end %rbf-kernel
