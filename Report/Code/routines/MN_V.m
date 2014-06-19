%____________________________________________
% Matrix Normal Variance Calculation         |
% Based on work of Sutherland et al.         |
%____________________________________________|
% Seb Grubb - sg3510@ic.ac.uk                |
%____________________________________________|
% This function returns a matrix of the      |
% estimated variance Var[u_i^Tv_j|R_O].      |
% As E[sigma^2] is ignored this is only      |
% useful for relative variance investigation.|
%====================Inputs==================|
% - U,V: feature matrices learnt from R      |
% - z_m: mask matrix, 1 for known value, 0   |
%        otherwise                           |
%====================Outputs=================|
% - R_var: matrix of same size as R with     |
%              values representing estimated |
%              sample variance based on input|
%              parameters.                   |
%____________________________________________|

function R_var = MN_V(U,V,z_m)
U_V = [U;V];
row_cov = cov(U_V');
dim_cov = cov(U_V);

[x,~] = size(U);
[y,d] = size(V);
R_var = zeros(x,y);
% Speed optimisation variables
E_ijk = zeros(x,y,d);
E_ijk_m = zeros(x,y,d);
% Attempt at optimisation - makes runtime worse
% E_ijkl = zeros(x,y,d,d);
% E_ijkl_m = zeros(x,y,d,d);
for i=1:x
    for j=1:y
        % E[Uki Vkj Uli Vlj] - E[Uki Vkj] E[Uli Vlj]
        % E[XaXb] = µaµb + Sigma(a,b.)
        if z_m(i,j) == 1
            for k=1:d
                for l=1:d
                    % Speed optimisation
                    if E_ijk_m(i,j,k) == 0
                        E_ijk_m(i,j,k) = 1;
                        E_ijk(i,j,k) = expect_norm2(U,V,row_cov,dim_cov,i,j,k);
                    end
                    if E_ijk_m(i,j,l) == 0
                        E_ijk_m(i,j,l) = 1;
                        E_ijk(i,j,l) = expect_norm2(U,V,row_cov,dim_cov,i,j,l);
                    end
	% Found to make performance worse
    %                 if E_ijkl_m(i,j,k,l) == 0
    %                     E_ijkl_m(i,j,k,l) = 1;
    %                     E_ijkl_m(i,j,l,k) = 1;
    %                     E_ijkl(i,j,k,l) = expect_norm4(U,V,row_cov,dim_cov,i,j,k,l);
    %                     E_ijkl(i,j,l,k) = E_ijkl(i,j,k,l);
    %                 end
                    R_var(i,j) = expect_norm4(U,V,row_cov,dim_cov,i,j,k,l) - E_ijk(i,j,k)*E_ijk(i,j,l);
	
	% Original non-optimised code
    %                 R_var(i,j) = expect_norm4(U,V,row_cov,dim_cov,i,j,k,l) - expect_norm2(U,V,row_cov,dim_cov,i,j,k)*expect_norm2(U,V,row_cov,dim_cov,i,j,l);
                end
            end
        end
    end
end
end