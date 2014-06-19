function [ R ] = matdownsample( R, val )
R_n = downsample(R',val);
R = downsample(R_n',val);
end

