%run kpmf
function [U,V,E] = kpmf_func(R,z_m,iter,num_feat,epsilon,diffu)
% Graph = R*R'/max(max(R*R'));
% Graph = Graph-diag(diag(Graph));
% Graph = Graph/max(max(Graph));
%Test mask graph
% Graph = z_m*z_m'/max(max(z_m*z_m'));
% Graph = Graph-diag(diag(Graph));
% Graph = Graph/max(max(Graph));
[L,M] = size(R);
% tmp_min = min(R(:));
% R = R - tmp_min;
% tmp_max = max(R(:));
% R = R/tmp_max;
sigma_r = .2;
gamma = 0.05;

% diagnal covariance matrix on movie side
% K_v = diffu * eye(M,M);
K_v = diffu*diff_ker(M,1.7);
K_v_inv = inv(K_v);


% K_u = diffu * eye(L,L);
% K_u_inv = inv(K_u);
% K_u = graphKernel(Graph, gamma);
K_u = diffu*diff_ker(L,1.75);
K_u_inv = pinv(K_u);

[U, V, E] = kpmf_gd1(R, z_m, num_feat, K_u_inv, K_v_inv, sigma_r, epsilon,iter);

end