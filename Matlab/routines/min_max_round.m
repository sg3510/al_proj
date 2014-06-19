function [ R ] = min_max_round( R,min_R,max_R )
R(R < min_R) = min_R;
R(R > max_R) = max_R;
R = round(R);
end

