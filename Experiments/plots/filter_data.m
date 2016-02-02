function data2 = filter_data(data)

tol = 1e-4;
dur = 60;

[tmp,N] = size(data);
take = ones(1,N);
last_take_x = -inf;
%{
for i = 1:N
	if( abs(last_take_y - data(2,i)) < tol  &&  data(1,i) - last_take_x < dur )
		take(i) = 0;
	else
		last_take_y = data(2,i);
		last_take_x = data(1,i);
	end
end
%}
for i=1:N
	if( data(1,i) > 100 && data(1,i) - last_take_x < 10 )
		take(i) = 0;
	elseif( data(1,i) > 1000 && data(1,i) - last_take_x < 100  )
		take(i) = 0;
	elseif( data(1,i) > 10000 && data(1,i) - last_take_x < 1000 )
		take(i)=0;
	else
		last_take_x = data(1,i);
	end
end

data2 = data(:,take==1);
