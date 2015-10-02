function product = kernel_inner_prod( param, model, x_f, y_f, ins_id, factor_id )

kv_cache_vect = model.kernel_cache{ins_id, factor_id};

product = 0.0;
for i = 1:model.alpha_size

	if( kv_cache_vect(i) ~= 0 )
		kv_diff = kv_cache_vect(i);
	else
		kv1 = kernel( param, model.patterns{i}, model.ytrues{i}, x_f, y_f );
		kv2 = kernel( param, model.patterns{i}, model.ystars{i}, x_f, y_f );
		kv_diff = kv1-kv2;
		
		kv_cache_vect(i) = kv_diff;
	end
	product = product + model.alpha(i) * kv_diff;
end

end %inner_prod

function kval = kernel(param, x1, y1, x2_f, y2_f)

num_dims = size(x1.data,1);
num_vars = size(x1.data,2);
num_states = x1.num_states;

kval = 0.0;
for i = 1:num_vars
	if( y1(i) == y2_f )
		kval = kval + kernel_factor(param, x1.data(:,i), x2_f);
	end
end

end %kernel

function kv = kernel_factor(param, x1f, x2f)

dif = (x1f-x2f);
norm_sq = sum(dif.*dif);
kv = exp(-param.gamma*norm_sq);

end %rbf-kernel
