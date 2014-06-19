function [U,V,e,e_val ] = pmf_func_val(R, z_m,R_val,z_m_val, iter,num_feat,lambda,epsilon)
    %% get features
    [x,y] = size(R);
    R = R.*z_m;
    %% model parameters
    %priors
    % mean_r = sum(sum(R))/samples;
    %remove mean
    % R = R - mean_r;
    R = R.*z_m;
    %latent matrix init
    U = 0.01*randn(x, num_feat); %randn
    V = 0.01*randn(y, num_feat);
    sample_no = sum(sum(z_m));
    sample_val = sum(sum(z_m_val));
    %% start training
    for step = 1:iter
%         step
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
        e_val(step) = sqrt(sum(sum((z_m_val.*(R_val-U*V').^2)))/sample_val);
        if (step>30)
            if (e_val(step) - e_val(step-1))> 0.001
%                 e_val(step) - e_val(step-1)
                fprintf(' loop ended at %d',step)
                break
            end
        end
    end
end

