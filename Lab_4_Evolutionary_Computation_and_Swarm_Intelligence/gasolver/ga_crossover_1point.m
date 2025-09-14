function [ offspring, fitness, dob, mob, parentIndices ] =...
    ga_crossover_1point( pop, rank, fitnessVals, pdob, pmob, eliteFitness, numOffspring,...
    gen, select_func, select_params, direction, obj_func, crossover_params)
%GA_CROSSOVER_1POINT Start of first parent + end of second parent
%   Discrete crossover of two parents.
%   Offspring start with the first genes from the first parent.
%   At a randomly chosen crossover point, switch to the second parent, take
%   all remaining genes from the end of that genome.
%   Gene positions are preserved (offspring's gene at position i is either
%   parent A at position i, or parent B at position i).
%   The crossover point is selected once each time the operation is
%   performed; the point will be different in each generation, but withn
%   that generation it is the same for all offspring.
    [popSize,chromLen] = size(pop);
    parentIndices = [...
        select_func(popSize,numOffspring,fitnessVals, rank, eliteFitness, direction, select_params) ;...
        select_func(popSize,numOffspring,fitnessVals, rank, eliteFitness, direction, select_params) ...
        ];

    % In the masks, a '1' means take value from parent 'A',
    % '0' means take from parent 'B'
    xmasks = zeros(numOffspring,chromLen);
    xmasks(:,1:randi(chromLen)) = 1;

    [offspring, fitness, dob, mob] = ga_x_linear(pop, parentIndices, xmasks, fitnessVals, pdob, pmob, numOffspring, gen, crossover_params);

end

