%____________________________________________
% Probabilistic Matrix Factorisation         |
% - Based on work of Ruslan Salakhutdinov and|
%   Andriy Mnih                              |
%____________________________________________|
% Seb Grubb - sg3510@ic.ac.uk                |
%____________________________________________|
% This function returns a matrix             |
% decomposition for a partially completed    |
% matrix in U and V variables.               |
%====================Inputs==================|
% - R: incomplete matrix to use for training |
% - z_m: mask matrix, 1 for known value, 0   |
%        otherwise                           |
% - iter: number of iterations               |
% - num_feat: number of latent features      |
% - lambda: regularisation parameter         |
% - epsilon: leanring rate                   |
%====================Outputs=================|
% - U,V: feature matrices, aim to reconstruct|
%        matrix by U'*V+mean_r = R_estimate  |
% - e: training error vector                 |
% - mean_r: mean value of R for known values |
%____________________________________________|
function [U,V,e, mean_r ] = pmf_func(R, z_m, iter,num_feat,lambda,epsilon)
	% get size
	[x,y] = size(R);
	%remove mean
	mean_r = sum(sum(R))/samples;
	R = R - mean_r;
	% Latent matrix initialisation
	U = 0.01*randn(x, num_feat);
	V = 0.01*randn(y, num_feat);
	sample_no = sum(sum(z_m));
	% start training
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
				end
			end
		end
		e(step) = sqrt(sum(sum((z_m.*(R-U*V').^2)))/sample_no);
	end
end

