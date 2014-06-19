function [ out ] = expect_norm4(U,V,row_cov,dim_cov,i,j,k,l)
% E[Uki Vkj Uli Vlj]
% E[XaXbXcXd] = µa µb µc µd + µc µd Sigma(ab) +µbµdSigma(ac) +µbµcSigma(ad)+µaµdSigma(bc)+µaµcSigma(bd)+µaµbSigma(cd) + Sigma(ab)Sigma(cd) + Sigma(ac)Sigma(bd) + Sigma(ad)Sigma(bc)
% a = i,k
% b = j,k
% c = i,l
% d = j,l
% Sigma(ij)Sigma(kl) = Sigma(ab)
x = length(U);
out = U(i,k)*V(j,k)*U(i,l)*V(j,l) + U(i,l)*V(j,l)* row_cov(i,j+x)*dim_cov(k,k) ... % µa µb µc µd + µc µd Sigma(ab)
        + V(j,k)*V(j,l) * row_cov(i,i)*dim_cov(k,l) ... %µbµdSigma(ac)
        + V(j,k)*U(i,l) * row_cov(i,j+x)*dim_cov(k,l) ... %µbµcSigma(ad)
        + V(j,k)*V(j,l) * row_cov(j+x,i)*dim_cov(k,l) ... %µa µd Sigma(bc)
        + U(i,k)*U(i,l) * row_cov(j+x,j+x)*dim_cov(k,l) ... %µa µc Sigma(bd)
        + U(i,k)*V(j,k) * row_cov(i,j+x)*dim_cov(l,l) ... %µaµbSigma(cd)
        + row_cov(i,j+x)*dim_cov(k,k) * row_cov(i,j+x)*dim_cov(l,l) ...%Sigma(ab)Sigma(cd)
        + row_cov(i,i)*dim_cov(k,l) * row_cov(j+x,j+x)*dim_cov(k,l) ...%Sigma(ac)Sigma(bd)
        + row_cov(i,j+x)*dim_cov(k,l) * row_cov(i,j+x)*dim_cov(k,l); %Sigma(ad)Sigma(bc)
end

