function [ offspring, fitness, dob, mob, parentIndices ] = ...
    ga_crossover_arithmetic( pop, rank, fitnessVals, pdob, pmob, eliteFitness, numOffspring,...
    gen, select_func, select_params, direction, obj_func, crossover_params)
%GA_CROSSOVER_ARITHMETIC Weighted average (float)
%   Generates offspring x with chromosome value at position i:
%
%       x_i = p1_i * weight(1) + p2_i * weight(2) + p3_1 * weight(3)...
%
%   for the set of weights defined by crossover.weight.
%   The length of crossover.weight determines the number of parents, with
%   one exception: a single weight is interpretted as requiring two
%   parents, where the resulting chromosome is:
%
%       x_i = p1_i * weight + p2_i * (1 - weight)
%
%   See also GA_CROSSOVER_GEOMETRIC, GA_CROSSOVER_BLEND
    
    persistent weight numParents;
    
    if gen == 0
        if any(strcmp('weight',fields(crossover_params)))
           if isscalar(crossover_params.weight)
               weight = [crossover_params.weight, 1 - crossover_params.weight];
           else
               weight = crossover_params.weight;
           end
        else
            weight = [.5 .5];
        end
        if sum(weight) ~= 1
           err = MException('GAsolver:InvalidInput','Input parameter outside expected range.');
           err = addCause(err, MException('GAsolver:BasWeights','Weight values do not sum to 1'));
           throw(err);
       end
       numParents = length(weight);
    end
    
    [popSize,chromLen] = size(pop);
    
    parentIndices = zeros(numParents, numOffspring);
    for i=1:numParents
        parentIndices(i,:) = select_func(popSize,numOffspring,fitnessVals, rank, eliteFitness, direction, select_params);
    end
    
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
    offspring = zeros(numOffspring,chromLen);
    for i=1:numParents
        offspring = offspring + (pop(parentIndices(i,:),:) * weight(i));
    end
    
end

