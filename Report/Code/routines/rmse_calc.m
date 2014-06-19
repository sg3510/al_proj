function [ rmse ] = rmse_calc( R_pred, R_test, z_test )
%RMSE Calculation
%   Calculates RMSE based on test set
sample_no = sum(z_test(:));
tmp = sum(sum(z_test.*(R_test - R_pred).^2));
rmse = sqrt(tmp/sample_no);
end

