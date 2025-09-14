function [f] = ga_fit_random(population, ~, ~)
% GA_FIT_RANDOM fitness values randomly generated in [0,1]

[popsize, dim] = size(population);
f = rand(1,popsize);
