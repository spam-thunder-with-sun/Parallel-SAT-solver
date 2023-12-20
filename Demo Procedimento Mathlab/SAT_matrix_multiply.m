clc;
close;
clear;

nLit = 3;
nClauses = 2;
problem = { 1 -3, 2 3 1};

clauses = [1 0 0 0 0 1; 1 1 1 0 0 0]';
solutions = zeros(2^nLit, 2*nLit);

%Riempio le soluzioni
for i = 1:2^nLit
    tmp = dec2bin(i-1, nLit);
    for y = 1:nLit
        solutions(i, nLit-y+1) = tmp(y)-'0';
        solutions(i, nLit-y+1 + nLit) = not(solutions(i, nLit-y+1));
    end
end

ris = solutions * clauses;
















