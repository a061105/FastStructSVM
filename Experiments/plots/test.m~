J = 35;
K = 1000000;

H = randn(K,J);
for k=1:K
	norm_2 = sqrt(H(k,:)*H(k,:)');
	H(k,:) = H(k,:) / norm_2;
end

h = H(1,:);

nnz( H(:,1) > 0 ) / K

%avg_j1_sq = mean(H(:,1).*H(:,1));
%avg_j1_sq
