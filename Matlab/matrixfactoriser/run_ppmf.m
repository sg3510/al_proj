function [ U,V,mv ] = run_ppmf(R_train, mask_train,k, R_v,mask_v )
%RUN_PPMF
%   Runs Parametric PMF
	[N,M]=size(R_train);
    inittype = 2;
    init.Lambda1=rand(k,N);
    init.Nu1=rand(k,N);
    init.Lambda2=rand(k,M);
    init.Nu2=rand(k,M);
    [mu1,Sigma1,mu2,Sigma2,tau,U,Nu1,V,Nu2,mv]=ppmfLearn(R_train,mask_train,R_v,mask_v,inittype,init);
    U = U'; V = V';
    % prediction on the whole matrix
%     [R_pred,rmse]=ppmfPred(U,V,mv,R_test,mask_test);
end

