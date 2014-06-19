function A = diff_ker(M,sigma)
l = zeros(2*M-1,1);
A = zeros(M,M);
for i=1:2*M-1
    l(i)=exp( -abs(i-M)^2/sigma^2);
end
for i=1:M
    A(:,i)=l(i:i+M-1);
end
A = rot90(A);
end