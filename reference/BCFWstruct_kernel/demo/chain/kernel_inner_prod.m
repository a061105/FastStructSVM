function product = kernel_inner_prod( param, model, x_f, y_f, ins_id, factor_id )

node_id = model.offset(ins_id) + factor_id;

alphas_y_f = model.alpha(:,y_f+1);

product = 0.0;
if( nnz(alphas_y_f) == 0 )
	return;
end

%product = model.kernel_matrix * alphas_y_f;

for i = find(alphas_y_f)'
	%kv = model.kernel_cache(i,node_id);
	%if( kv == -1 )
		kv = kernel( param, model.patterns{i}, x_f );
		%model.kernel_cache(i,node_id) = kv;
	%end
	product = product + alphas_y_f(i) * kv;
end


end %inner_prod

