function [mu1,Sigma1,mu2,Sigma2,tau]=ppmfMstep(R,mask,Lambda1,Nu1,Lambda2,Nu2) 
% 
% Author: Hanhuai Shan. 04/2012.  
%
% M step of ppmf
%   
%   k:          the rank after decomposition
%   M:          #movies
%   N:          #users
%
% Input:
%   R:          N*M, rating matrix for learning
%   mask:       N*M, indicator matrix for R, 1 is non-missing entry
%   Lambda1:    k*N
%   Nu1:        k*M
%   Lambda2:    k*N
%   Nu2:        k*M
%
% Output:
%   mu1:        k*1
%   Sigma1:     k*k
%   mu2:        k*1
%   Sigma2:     k*k
%   tau:        scaler
%-----------------------------------------------------------------

[N,M]=size(R);
[k,N]=size(Lambda1);


% Update mu1
mu1=sum(Lambda1,2)/N;

% Update mu2
mu2=sum(Lambda2,2)/M;

% Sigma1
temp=0;
for i=1:N
    temp=temp+diag(Nu1(:,i))+(Lambda1(:,i)-mu1)*(Lambda1(:,i)-mu1)';
end
Sigma1=temp/N;
Sigma1=Sigma1+exp(-30);

% Sigma2
temp=0;
for j=1:M
    temp=temp+diag(Nu2(:,j))+(Lambda2(:,j)-mu2)*(Lambda2(:,j)-mu2)';
end
Sigma2=temp/M;
Sigma2=Sigma2+exp(-30);

%tau
tau=sum(sum((R.^2-2*R.*(Lambda1'*Lambda2)...
    +(Lambda1'*Lambda2).^2 ...
    +Lambda1'.^2*Nu2+Nu1'*(Lambda2.^2)+Nu1'*Nu2).*mask))/sum(sum(mask));

