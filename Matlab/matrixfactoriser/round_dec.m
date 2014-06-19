function [ out ] = round_dec( in, dec)
%ROUND_DEC Rounds variable to 10^dec
   dec = 10^dec;
   out = round(in*dec)/dec;

end

