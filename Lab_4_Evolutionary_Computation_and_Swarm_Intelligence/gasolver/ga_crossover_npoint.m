function [ offspring, fitness, dob, mob, parentIndices ] =...
    ga_crossover_npoint( pop, rank, fitnessVals, pdob, pmob, eliteFitness, numOffspring,...
    gen, select_func, select_params, direction, obj_func, crossover_params)
%GA_CROSSOVER_NPOINT Discrete combination with n crossover points
%   Generalized form of 1-point crossover.
%
%   Offspring start with first genes from parent A. At a random point,
%   switch to genes from parent B, then later switch back to A,
%   and so on. There will be (approximately) n switches.
%   The switch points are randomly chosen in each generation, but within
%   the same generation all switches occur at the same points.

    persistent localParam;
    if gen == 0
        localParam = ga_param_defaults(crossover_params, 'n', 2);
    end
    
    [popSize,chromLen] = size(pop);
    parentIndices = [...
        select_func(popSize,numOffspring,fitnessVals, rank, eliteFitness, direction, select_params) ;...
        select_func(popSize,numOffspring,fitnessVals, rank, eliteFitness, direction, select_params) ...
        ];
    
    % make xmasks
    ind = randperm(chromLen-1) + 1;
    region = [1 sort(ind(1:localParam.n))];
    xmasks = zeros(numOffspring,chromLen);
    for i=2:2:localParam.n+1
        xmasks(:,region(i-1):region(i)) = 1;
    end
    
    [offspring, fitness, dob, mob] =...
        ga_x_linear(pop, parentIndices, xmasks, fitnessVals, pdob, pmob, numOffspring, gen, crossover_params);
end

