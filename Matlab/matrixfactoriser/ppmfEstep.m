function [Lambda1_t,Nu1_t,Lambda2_t,Nu2_t]=ppmfEstep(R,mask,mu1,Sigma1,mu2,Sigma2,tau,Lambda1,Nu1,Lambda2,Nu2)
% 
% Author: Hanhuai Shan. 04/2012.  
%
% E-step of ppmf
%
%   k:          the rank after decomposition
%   M:          #movies
%   N:          #users
%
% Input:
%   --data--
%   R:          N*M, rating matrix for learning
%   mask:       N*M, indicator matrix for R, 1 denotes non-missing entry, and 0 denotes missing entry
%   steps:      steps for the E step
%   --model parameters--
%   mu1:        k*1
%   Sigma1:     k*k
%   mu2:        k*1
%   Sigma2:     k*k
%   tau:        scaler
%   --initializations for variational parameters in the current E step--
%   Lambda1:    k*N
%   Nu1:        k*M
%   Lambda2:    k*N
%   Nu2:        k*M
%
% Output:
%   Lambda1_t:  k*N
%   Nu1_t:      k*M
%   Lambda2_t:  k*N
%   Nu2_t:      k*M
%---------------------------------------------------------------------


k=length(mu1);
[N,M]=size(R);

Lambda1_t=Lambda1;
Nu1_t=Nu1;
Lambda2_t=Lambda2;
Nu2_t=Nu2;

invSigma1=inv(Sigma1);
invSigma2=inv(Sigma2);
invSigma1mu1=invSigma1*mu1;
invSigma2mu2=invSigma2*mu2;


t=1;
steps=10;

% We run the iterations for fixed times of steps.
% Alternatives would be tracking the change of log-likelihood, or the
% change of the variational parameters, etc..
while t<steps
    
    %update Lambda1
    for j=1:M
        Lambda2_t_square_temp(:,:,j)=Lambda2_t(:,j)*Lambda2_t(:,j)'/tau;
    end
    Nu2_t_temp=Nu2_t*mask'/tau;
    right_temp=invSigma1mu1*ones(1,N)+Lambda2_t*(R.*mask)'/tau;
    
    
    for i=1:N
        J=find(mask(i,:)==1);
        temp=sum(Lambda2_t_square_temp(:,:,J),3);

        left=invSigma1+temp+diag(Nu2_t_temp(:,i));
        right=right_temp(:,i);
        Lambda1_tt(:,i)=inv(left)*right;   
    end
    
    % Update Lambda2
    for i=1:N
        Lambda1_tt_square_temp(:,:,i)=Lambda1_tt(:,i)*Lambda1_tt(:,i)'/tau;
    end   
    Nu1_t_temp=Nu1_t*mask/tau;
    right_temp=(invSigma2mu2)*ones(1,M)+Lambda1_tt*(R.*mask)/tau;
 
    for j=1:M
        I=find(mask(:,j)==1);
        temp=sum(Lambda1_tt_square_temp(:,:,I),3);
        left=invSigma2+temp+diag(Nu1_t_temp(:,j));
        right=right_temp(:,j);
        Lambda2_tt(:,j)=inv(left)*right;
    end
    
    % Update Nu1
    Nu1_tt=1./((Lambda2_tt.^2+Nu2_t)*mask'/tau+diag(invSigma1)*ones(1,N));
    
    % Update Nu2
    Nu2_tt=1./((Lambda1_tt.^2+Nu1_tt)*mask/tau+diag(invSigma2)*ones(1,M));
   
    % set up for the next iteration
    Lambda1_t=Lambda1_tt;
    Lambda2_t=Lambda2_tt;
    Nu1_t=Nu1_tt;
    Nu2_t=Nu2_tt;
    
    t=t+1;
   
end

