function [ parentIndices ] = ga_select_random( popSize, numOffspring, ~, varargin )
%GA_SELECT_RANDOM Radnom selection using the uniform distribution
%   This is really intended only as a baseline.
    
    parentIndices=randi(popSize, 1, numOffspring); 
end

