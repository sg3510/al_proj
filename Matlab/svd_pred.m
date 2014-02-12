clc;
clear all;
load ibn_sina
A = [[5 5 4 3 5 4 3 2 1 5]
     [5 5 4 3 5 4 3 2 1 5]
     [5 5 4 3 5 4 3 2 1 5]
     [5 5 4 3 5 4 3 2 1 5]
     [1 1 2 3 5 2 3 4 5 1]
     [1 1 2 3 5 2 3 4 5 1]
     [1 1 2 3 5 2 3 4 5 1]
     [1 1 2 3 5 2 3 4 5 1]
     [5 5 2 1 5 5 2 1 2 5]
     [5 5 2 1 5 5 2 1 2 5]
     [5 5 2 1 5 5 2 1 2 5]
     [5 5 2 1 5 5 2 1 2 5]
     [5 5 2 1 5 5 2 1 2 5]
     [2 2 5 5 5 1 1 2 3 5]
     [2 2 5 5 5 1 1 2 3 5]
     [2 2 5 5 5 1 1 2 3 5]
     [2 2 5 5 5 1 1 2 3 5]
     [2 2 5 5 5 1 1 2 3 5]
     [2 2 5 5 5 1 1 2 3 5]
     [2 2 5 5 5 1 1 2 3 5]
     [2 2 5 5 5 1 1 2 3 5]
     [2 2 5 5 5 1 1 2 3 5]
     [2 2 5 5 5 1 1 2 3 5]];

 z_m = zeros(size(A));

 [x,y] = size(A);
for i=1:(100000 + 30*randn(1))
    a = randi(x);
    b = randi(y);
    while(z_m(a,b)==1)
        a = randi(x);
        b = randi(y);
    end
    z_m(a,b) = 1;     
end
fprintf('known values:%d\n', sum(sum(abs(z_m))))
fprintf('Or %2.1f %%\n', sum(sum(abs(z_m)))/(x*y)*100 )
 B = A.*z_m;
 z_m_i = ones(size(A))-z_m;
 %make 4 feature decomposition
 features = 8;
 lrate = 0.0003;
 iter = 40;
 U = ones(x,features);
 V = ones(features,y);
 Z = U*V;
 for f = 1:features
     for a = 1:iter
         Z=U*V;
         for i = 1:x
             for j = 1:y
                 %Z=U*V;
                 if (z_m(i,j) == 1)
                     err = A(i,j) - Z(i,j);
                     uv = U(i,f);
                     U(i,f) = U(i,f) + V(f,j)*lrate*err;
                     V(f,j) = V(f,j) + uv*lrate*err;
                 end
             end
         end
         error(a,f) = sum(sum(abs(A-U*V)));
         if (a>10)
             if (error(a,f) - error(a-1,f) > 0.1)
                 error(a+2:end,f) = error(a,f);
                 error(a+1,f) = -1;
                 break
             end
         end
     end
 end
 err_d = sum(sum(abs(A-round(U*V)).*z_m_i));
 corr = sum(sum(z_m_i))-sum(sum(abs(A-round(U*V)).*z_m_i));
%  fprintf('error:%d\n', err_d)
%  fprintf('correct:%2.1f%%\n', 100*corr/(corr+err_d))

plot(error)
% round((U*V).*z_m_i)
 