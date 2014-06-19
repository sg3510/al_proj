function [R_pred,rmse]=ppmfPred(U,V,mv,R_test,mask_test)
% 
% Author: Hanhuai Shan. 04/2012.  
% 
% Predict on the test set
%
%   k:          the rank after decomposition
%   M:          #movies
%   N:          #users
%
% Input:
%   --- factorization results from the learning process---
%   U:    k*N
%   V:    k*M
%   mv:   mean of all non-missing entries in R_train
%   ---data---
%   R_test:          N*M, rating matrix for test
%   mask_test:       N*M, indicator matrix for R_test, 1 is non-missing entry
%
%
% Output:
%   R_pred:         Result for prediciton
%   rmse:           RMSE
%----------------------------------------------------------------------

R_pred=(U'*V+mv).*mask_test;
rmse=sqrt(sum(sum((R_pred-R_test).^2.*mask_test))/sum(sum(mask_test)));