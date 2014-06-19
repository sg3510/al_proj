clear all
clc
close all

load synthetic_data
% R = data;
%% initial random sampling
[x,y] = size(R);
%select 10% of data
samples = round(0.1*x*y);
z_m = zeros(x,y);
%loop until #samples is done.
for i=1:samples
    a = randi(x);
    b = randi(y);
    %ensure no duplicate
    while(z_m(a,b)==1)
        a = randi(x);
        b = randi(y);
    end
    z_m(a,b) = 1;     
end
R_orig = R;
R = R.*z_m;
%% model parameters
%good params iter = 200; feat = 5; 
%looping
iter = 200;
%batches = 10; only useful for mega large
%model vals
num_feat = 10;
lambda  = .05; % Regularization parameter
epsilon= 0.005; % Learning rate 
%priors
% mean_r = sum(sum(R))/samples;
mean_r = 0;
%remove mean
% R = R - mean_r;
R = R.*z_m;
%latent matrix init
U = 0.01*randn(x, num_feat);
V = 0.01*randn(y, num_feat);

%% start training
for step = 1:iter
    for i = 1:x
        for j = 1:y
            if z_m(i,j) == 1
                eij = R(i,j) - U(i,:)*V(j,:)';
                if eij > 100000
                    eij
                end
                for k=1:num_feat
                    U(i,k) = U(i,k) + epsilon * (2*eij * V(j,k) - lambda * U(i,k));
                    V(j,k) = V(j,k) + epsilon * (2*eij * U(i,k) - lambda * V(j,k));
                end
                lambda = lambda - epsilon*lambda*(norm(U,'fro')+norm(V,'fro'));
                lam(step) = lambda;
            end
        end
    end
    e(step) = sqrt(sum(sum((z_m.*(R-U*V').^2))));
    R_h = U*V' + mean_r;
    R_h(R_h < 1) = 1;
    R_h(R_h > 5) = 5;
    real_e(step) = sqrt(sum(sum(((1-z_m).*(R_orig-R_h).^2))));
end
e(end)
real_e(end)
%% plot data
R_h = U*V' + mean_r;
R_h(R_h < 1) = 1;
R_h(R_h > 5) = 5;
R = round(R);
R_h = round(R_h);
figure(1)
image(R_h/max(max(R_h))*50)
figure(2)
image(R_orig/max(max(R_orig))*50)
figure(3)
image(z_m.*R_orig/max(max(R_orig))*50)
figure(4)
plot([real_e; e]')