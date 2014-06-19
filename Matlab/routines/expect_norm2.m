function [ out ] = expect_norm2(U,V,row_cov,dim_cov,i,j,k)
%E[XaXb] = µaµb + ?a,b. 
%E[Uki Vkj]
%?a,b = ?ij?kl
% fprintf('i:%d - j:%d \n',i,j+length(U))
out = U(i,k)*V(j,k) + row_cov(i,j+length(U))*dim_cov(k,k);
end

