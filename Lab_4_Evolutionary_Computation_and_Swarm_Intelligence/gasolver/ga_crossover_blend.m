function [ offspring, fitness, dob, mob, parentIndices ] = ...
    ga_crossover_blend( pop, rank, fitnessVals, pdob, pmob, eliteFitness, numOffspring,...
    gen, select_func, select_params, direction, obj_func, crossover_params)
%GA_CROSSOVER_BLEND Stochastic weighted average (float)
%   Generates offspring x with chromosome value at position i:
%
%       x_i = p1_i * gamma + p2_i * (1 - gamma)
%
%   where gamma = (1 + 2 * weight(1)) * U - wieght(1),
%   and U is randomly generated in [0,1].
%   The number of parents is fixed at 2,
%   and weights other than weight(1) are ignored.
%
%   See also GA_CROSSOVER_GEOMETRIC, GA_CROSSOVER_ARITHMETIC
    
    persistent weight;
    
    if gen == 0
        if any(strcmp('weight',fields(crossover_params)))
           if isscalar(crossover_params.weight)
               weight = crossover_params.weight;
           else
               weight = crossover_params.weight(1);
           end
        else
            weight = 0.5;
        end
    end
    
    [popSize,chromLen] = size(pop);
    
    parentIndices = zeros(2, numOffspring);
    for i=1:2
        parentIndices(i,:) = select_func(popSize,numOffspring,fitnessVals, rank, eliteFitness, direction, select_params);
    end
    
    gamma = rand(numOffspring,chromLen) * (1 + 2*weight) - weight;
    reprodIndices=or(   rand(numOffspring,1)<1-crossover_params.prob,...
                        (max(parentIndices,[],1) == min(parentIndices,[],1))');
                        %parentIndices(1,:)' == parentIndices(2,:)');
                    
   % save fitness vals for preserved parents:
    fitness = ones(1,numOffspring) * -inf;
    fitness(reprodIndices) = fitnessVals(parentIndices(1,reprodIndices));
    dob = gen * ones(1,numOffspring);
    dob(reprodIndices) = pdob(parentIndices(1,reprodIndices));
    mob = ones(1,numOffspring);
    mob(reprodIndices) = pmob(parentIndices(1,reprodIndices));
    % do the crossover
    offspring = min(pop(parentIndices(1,:),:),pop(parentIndices(2,:),:)) .* (1 - gamma) ...
        + max(pop(parentIndices(1,:),:),pop(parentIndices(2,:),:)) .* gamma;
end

