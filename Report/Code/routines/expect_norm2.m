function [ out ] = expect_norm2(U,V,row_cov,dim_cov,i,j,k)
	%E[XaXb] = µaµb + Sigma(a,b). 
	%E[Uki Vkj]
	% Sigma(a,b) = Sigma(i,j)Sigma(k,l)
	out = U(i,k)*V(j,k) + row_cov(i,j+length(U))*dim_cov(k,k);
end

