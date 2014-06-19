%____________________________________________
% Correlated matrix creation                 |
%____________________________________________|
% This function creates a matrix with base   |
% row profiles (i.e. users) and randomly     |
% generates their preferences. The remaining |
% users are a mixture of the base type users.|
%____________________________________________|

clear all
clc

% types of users
user_t = 3;
% Rank of matrix
rank = 3;
% total number of users
users = 9;
% movies
columns = 16;
% Randomly generate
% round_dec(x,1) rounds number to first decimal
% rand_dual is a two ended Gaussian random generator.
U = round_dec(rand_dual(user_t,rank),1);
V = round_dec(rand_dual(rank,columns),1);
% Loop through base users and create correlations
for i=user_t+1:users
	% randomly choose association
    profile = randsample(1:user_t,randsample(2:user_t-1,1));
    k = length(profile);
	% correlation probabilities
    prob = zeros(k,1);
    for j=1:k
        if (j ~= 1) && (j ~= k);
            prob(j) = round_dec((1 - sum(prob))*rand(1),1);
        end
        if j == 1;
            prob(j) = round_dec(rand(1),1);
        end
        if j == k;
            prob(j) = 1 - sum(prob);
        end
    end
    for j=1:k
        U(i,:) =  prob'*U(profile,:);
    end
end
% Plot to check
R = U*V;
R = round(R/max(max(R))*5);
R(R==0) = 5;
hist(R(:))
figure(2)
image(R/max(max(R))*50)