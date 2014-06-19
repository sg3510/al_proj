function [ out ] = rand_dual( x,y)
%RAND_DUAL Two ended Gaussian Generator
%   Two ended Gaussian constrained between 0 and 1
    out = 0.25*(randn(x,y));
    indices = find(out < 0);
    out(indices) = 1 - out(indices);
    indices = find(out > 1);
    out(indices) = 1-abs(out(indices)-1);
    out(out < 0) = 0;
    out(out > 1) = 1;
end

