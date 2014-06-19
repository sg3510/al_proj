clear

load data
[N,M]=size(R_train);

k=5;

% set the init type
inittype=2;


% perform ppmf given the initial value for variational parameters
if inittype==2;
    init.Lambda1=rand(k,N);
    init.Nu1=rand(k,N);
    init.Lambda2=rand(k,M);
    init.Nu2=rand(k,M);
    [mu1,Sigma1,mu2,Sigma2,tau,Lambda1,Nu1,Lambda2,Nu2,mv]=ppmfLearn(R_train,mask_train,R_v,mask_v,inittype,init);
    % prediction on the whole matrix
    [R_pred,rmse]=ppmfPred(Lambda1,Lambda2,mv,R_test,mask_test);
% perform ppmf given the initial value for model parameters
elseif inittype==1
    init.mu1=rand(k,1);
    init.Sigma1=rand(k,k);
    init.mu2=rand(k,1);
    init.Sigma2=rand(k,k);
    init.tau=1;
    [mu1,Sigma1,mu2,Sigma2,tau,Lambda1,Nu1,Lambda2,Nu2,mv]=ppmfLearn(R_train,mask_train,R_v,mask_v,inittype,init);
    % prediction on the whole matrix
    [R_pred,rmse]=ppmfPred(Lambda1,Lambda2,mv,R_test,mask_test);
end

