function [ offspring, fitness, dob, mob, indices ] = ...
    ga_crossover_none( pop, rank, fitnessVals, pdob, pmob, eliteFitness, numOffspring,...
    gen, select_func, select_params, direction, obj_func, crossover_params)
%GA_CROSSOVER_NONE Offspring are copies of selected parents
%   Use this only if you want to disable crossover.
%   Setting the crossover probability to 0 would have the same result, but
%   using this function should be slightly faster.
    
    [popSize,chromLen] = size(pop);
    indices = select_func(popSize,numOffspring,fitnessVals, rank, eliteFitness, direction, select_params);
    fitness = fitnessVals(indices);
    dob = pdob(indices);
    mob = pmob(indices);
    offspring = pop(indices,:);
end

