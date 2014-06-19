function [mu1,Sigma1,mu2,Sigma2,tau,Lambda1_t,Nu1_t,Lambda2_t,Nu2_t,mv]=ppmfLearn(R_train,mask_train,Rv,maskv,inittype,init)
% 
% Author: Hanhuai Shan. 04/2012.  
%
% Learn ppmf by runing variational EM
%
%   k:          the rank after decomposition
%   M:          #movies
%   N:          #users
%
% Input:
%   ---data---
%   R_train:    N*M, rating matrix for training
%   mask_train: N*M, indicator matrix for R_train, 1 is non-missing entry
%   Rv:         N*M, rating matrix for valdation to determin early stopping
%   maskv:      N*M, indicator matrix for RV
%   ---initialization---
%   inittype:   if inittype==1, model parameters are initialized; 
%               if inittype==2, variational parameters are initialized;
%   init:    
%   if inittype ==1, then init contains:
%   ---initialization for model parameters---
%   mu1:        k*1
%   Sigma1:     k*k
%   mu2:        k*1
%   Sigma2:     k*k
%   tau:        scaler
%   if inittype==2, then init contains:
%   --- initialization for variational parameters---
%   Lambda1:    k*N
%   Nu1:        k*N
%   Lambda2:    k*M
%   Nu2:        k*M
%
% Output:
%   mu1_t:        k*1
%   Sigma1_t:     k*k
%   mu2_t:        k*1
%   Sigma2_t:     k*k
%   tau_t:        scaler
%   Lambda1:      k*N U
%   Nu1:          k*N
%   Lambda2:      k*M V
%   Nu2:          k*M
%   mv:           mean of all non-missing entries in R_train
%----------------------------------------------------------------------


[N,M]=size(R_train);
mv=sum(sum(R_train.*mask_train))/sum(sum(mask_train)); % mean of all non-missing entries
R_train=(R_train-mv).*mask_train;

epsilon=0.001;
steps=500;
t=1;
valError_t=100;
minstep=5;

% if only given the model parameters, randomly initialize the variational
% parameters
if inittype==1
    mu1=init.mu1;
    Sigma1=init.Sigma1;
    mu2=init.mu2;
    Sigma2=init.Sigma2;
    tau=init.tau;
    k=length(mu1);
    Lambda1_t=0.1*rand(k,N);
    Nu1_t=rand(k,N);
    Lambda2_t=0.1*rand(k,M);
    Nu2_t=rand(k,M);
% if only given the variational parameters, perform an E-step first to get
% the model parameters
elseif inittype==2
    Lambda1_t=init.Lambda1;
    Lambda2_t=init.Lambda2;
    Nu1_t=init.Nu1;
    Nu2_t=init.Nu2;
    [mu1,Sigma1,mu2,Sigma2,tau]=ppmfMstep(R_train,mask_train,Lambda1_t,Nu1_t,Lambda2_t,Nu2_t);
    k=length(mu1);
else
    disp('input error.')
end

while t<steps %&& e>epsilon 
    % E-step
    [Lambda1_tt,Nu1_tt,Lambda2_tt,Nu2_tt]=ppmfEstep(R_train,mask_train,mu1,Sigma1,mu2,Sigma2,tau,Lambda1_t,Nu1_t,Lambda2_t,Nu2_t);
     
    % M-step   
    [mu1,Sigma1,mu2,Sigma2,tau]=ppmfMstep(R_train,mask_train,Lambda1_tt,Nu1_tt,Lambda2_tt,Nu2_tt);
        
    % error for validation
    valError_tt=sqrt(sum(sum((Lambda1_tt'*Lambda2_tt+mv-Rv).^2.*maskv))/sum(sum(maskv)));
    e=(valError_t-valError_tt)/valError_t; % difference of validation error from the last iteration
%     disp(['t=',int2str(t),' rmse= ',num2str(full(valError_tt))]);    
    
    % setup for the next iteration
    Lambda1_t=Lambda1_tt;
    Lambda2_t=Lambda2_tt;
    Nu1_t=Nu1_tt;
    Nu2_t=Nu2_tt;
    valError_t=valError_tt;
    t=t+1;
    
    % After minstep iterations, if the validation error stops decreasing in
    % any iteration, finish.
    if e<epsilon && t>=minstep
        break;
    end
    
end

