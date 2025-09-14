function [ offspring, fitness, dob, mob, parentIndices ] = ...
    ga_crossover_uniform( pop, rank, fitnessVals, pdob, pmob, eliteFitness, numOffspring,...
    gen, select_func, select_params, direction, obj_func, crossover_params)
%GA_CROSSOVER_UNIFORM Randomly choose genes from the two parents
%   Discrete crossover from two parents: at each position, the offspring
%   gets the value at that position from a randomly selected parent.
    
    persistent localParam uniformCrossmaskRepos;
    [popSize,chromLen] = size(pop);
    parentIndices = [...
        select_func(popSize,numOffspring,fitnessVals, rank, eliteFitness, direction, select_params) ;...
        select_func(popSize,numOffspring,fitnessVals, rank, eliteFitness, direction, select_params) ...
        ];
    if gen == 0
        localParam = ga_param_defaults(crossover_params, 'maskReposFactor',5,'useRepositories',true);
        uniformCrossmaskRepos=rand(numOffspring,(chromLen+1)*localParam.maskReposFactor)<0.5;
    end
    if localParam.useRepositories
        temp=floor(rand*chromLen*(localParam.maskReposFactor-1));
        xmasks=uniformCrossmaskRepos(1:numOffspring,temp+1:temp+chromLen);
    else
        xmasks=rand(numOffspring, chromLen)<.5;
        xmasks = +xmasks; % remove 'logicalness' of array
    end
    
    [offspring, fitness, dob, mob] = ga_x_linear(pop, parentIndices, xmasks, fitnessVals, pdob, pmob, numOffspring, gen, crossover_params);
end

