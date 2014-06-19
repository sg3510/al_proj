function [ R_train,z_train, R_test, z_test,a,b] = random_sample( R_train,z_train, R_test, z_test)
%Random Sample 
%   Randomly Samples from test set

% Get all positive indices in R_test
[a,b] = find(z_test == 1);
j = randi(length(a));
a = a(j);
b = b(j);

%Swap
R_train(a,b) = R_test(a,b);
z_train(a,b) = 1;
R_test(a,b) = 0;
z_test(a,b) = 0;
end

