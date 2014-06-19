function [ out ] = rand_dual( x,y)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    out = 0.25*(randn(x,y));
    indices = find(out < 0);
    out(indices) = 1 - out(indices);
    indices = find(out > 1);
    out(indices) = 1-abs(out(indices)-1);
    out(out < 0) = 0;
    out(out > 1) = 1;
end

