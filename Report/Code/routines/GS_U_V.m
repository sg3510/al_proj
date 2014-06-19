function [U,V,e,mean_rating] = GS_U_V(R,z_m,U,V)
epoch=1; 
beta = 3.5;
iter = 4;
maxepoch=iter; 
[x,y] = size(R);
[~,num_feat ] = size(U);
% Initialize hierarchical priors 
% beta=2; % observation noise (precision) 
mu_u = zeros(num_feat,1);
mu_v = zeros(num_feat,1);
alpha_u = eye(num_feat);
alpha_v = eye(num_feat);  

% parameters of Inv-Whishart distribution (see paper for details) 
WI_u = eye(num_feat);
b0_u = beta;
df_u = num_feat;
mu0_u = zeros(num_feat,1);

WI_v = eye(num_feat);
b0_v = beta;
df_v = num_feat;
mu0_v = zeros(num_feat,1);
%get sample #
sample_no = sum(sum(z_m));
mean_rating = sum(sum(z_m.*R))/sample_no;



  % Initialization using MAP solution found by PMF. 
%% Do simple fit
mu_u = mean(U)';
d=num_feat;
alpha_u = inv(cov(U));

mu_v = mean(V)';
alpha_v = inv(cov(V));%U

R=R';



for epoch = epoch:maxepoch

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Sample from movie hyperparams (see paper for details)  
  N = size(V,1);
  x_bar = mean(V)'; 
  S_bar = cov(V); 

  WI_post = inv(inv(WI_v) + N/1*S_bar + ...
            N*b0_v*(mu0_v - x_bar)*(mu0_v - x_bar)'/(1*(b0_v+N)));
  WI_post = (WI_post + WI_post')/2;

  df_mpost = df_v+N;
  alpha_v = wishrnd(WI_post,df_mpost);   
  mu_temp = (b0_v*mu0_v + N*x_bar)/(b0_v+N);  
  lam = chol( inv((b0_v+N)*alpha_v) ); lam=lam';
  mu_v = lam*randn(num_feat,1)+mu_temp;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Sample from user hyperparams
  N = size(U,1);
  x_bar = mean(U)';
  S_bar = cov(U);

  WI_post = inv(inv(WI_u) + N/1*S_bar + ...
            N*b0_u*(mu0_u - x_bar)*(mu0_u - x_bar)'/(1*(b0_u+N)));
  WI_post = (WI_post + WI_post')/2;
  df_mpost = df_u+N;
  alpha_u = wishrnd(WI_post,df_mpost);
  mu_temp = (b0_u*mu0_u + N*x_bar)/(b0_u+N);
  lam = chol( inv((b0_u+N)*alpha_u) ); lam=lam';
  mu_u = lam*randn(num_feat,1)+mu_temp;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Start doing Gibbs updates over user and 
  % movie feature vectors given hyperparams.  

  for gibbs=1:4 
%     fprintf(1,'\t\t Gibbs sampling %d \r', gibbs);

    %%% Infer posterior distribution over all movie feature vectors 
    R=R';
    z_m = z_m';
    for mm=1:y
%        fprintf(1,'movie =%d\r',mm);
%        ff = find(z_m(:,mm)==1);
       ff = find(z_m(mm,:)==1);
       MM = U(ff,:);
       rr = R(ff,mm)-mean_rating;
       covar = inv((alpha_v+beta*MM'*MM));
       mean_m = covar * (beta*MM'*rr+alpha_v*mu_v);
       lam = chol(covar); lam=lam'; 
       V(mm,:) = lam*randn(num_feat,1)+mean_m;
     end

    %%% Infer posterior distribution over all user feature vectors 
     R=R';
     z_m = z_m';
     for uu=1:x
%        fprintf(1,'user  =%d\r',uu);
       %ff = find(z_m(:,uu)==1);
       ff = find(z_m(uu,:)==1);
       MM = V(ff,:);
       rr = R(ff,uu)-mean_rating;
       covar = inv((alpha_u+beta*MM'*MM));
       mean_u = covar * (beta*MM'*rr+alpha_u*mu_u);
       lam = chol(covar); lam=lam'; 
       U(uu,:) = lam*randn(num_feat,1)+mean_u;
     end
  end 
  R_h = U*V' + mean_rating;
  tmp = sum(sum(z_m.*(R' - R_h).^2));
  err = sqrt(tmp/sample_no);
%   fprintf(1, '\nEpoch %d \t Average Test RMSE %6.4f \n', epoch, err);
  e(epoch) = err;

end 
end

