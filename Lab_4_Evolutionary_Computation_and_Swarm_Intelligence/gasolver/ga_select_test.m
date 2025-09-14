function [ parentIndices ] = ga_select_test( popSize, numOffspring, fitnessVals, varargin )
%GA_SELECT_TEST Selects same parent for every offspring
%   Intended for use in plotting the distribution of offspring that result
%   from a stochastic crossover method. 
%
%   Each time the function is called, it returns a vector consisting of
%   multiple copies of the index of the next member of the population.
%   First call returns [1, 1, 1, 1, ...]; next call [2, 2, 2, ...]. When
%   the index exceeds the size of the sample population, resets to 1.

    persistent index;
    if isempty(index)
        index = 0;
    else
        index = mod(index + 1,popSize);
    end
    parentIndices = ones(1,numOffspring) + index;

end

