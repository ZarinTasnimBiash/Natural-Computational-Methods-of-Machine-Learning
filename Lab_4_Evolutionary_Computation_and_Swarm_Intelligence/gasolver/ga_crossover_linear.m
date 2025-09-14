function [ offspring, fitness, dob, mob, parentIndices ] = ...
    ga_crossover_linear( pop, rank, fitnessVals, pdob, pmob, eliteFitness, numOffspring,...
    gen, select_func, select_params, direction, obj_func, crossover_params)
%GA_CROSSOVER_LINEAR Extrapolating, 2 parent crossover (real)
%
%   One of the earliest crossover operators for real valued chromosomes, by
%   Wright. Also the simplest extrapolating real-valued crossover (i.e., a
%   crossover that can produce offspring from outside the range defined by
%   the parent values).
%
%   For each pair of parents, three offspring are created:
%
%       x_1 = weight * p_1 + (1-weight) * p_2
%       x_2 = (1 + weight) * p_1 - weight * p_2
%       x_3 = (1 + weight) * p_2 - weight * p_1
%
%   If x_2 or x_3 is out of bounds, it is discarded; otherwise, the two
%   offspring with the highest fitness are kept. Note that this requirement
%   results in some additional fitness function evaluations (but not too
%   many --- fitness values calculated here are retained, and only need to
%   be recalculated later if the offspring is mutated in this generation).
%
%   0.5 is the default 'weight' value, and is the value used in the
%   original paper, but other values in the range (0,1) make sense also.
%
%   See also GA_CROSSOVER_ARITHMETIC
    persistent weight bounds;
    
    [popSize,chromLen] = size(pop);
    
    numPairs = 1 + numOffspring / 2;
    
    parentIndices = [...
        select_func(popSize,numPairs,fitnessVals, rank, eliteFitness, direction, select_params) ;...
        select_func(popSize,numPairs,fitnessVals, rank, eliteFitness, direction, select_params) ...
        ];
    
    if gen == 0
        if any(strcmp('weight',fields(crossover_params)))
           weight = crossover_params.weight(1);
        else
           weight = .5;
        end
        if weight <= 0 || weight >=1
           err = MException('GAsolver:InvalidInput','Input parameter outside expected range.');
           err = addCause(err, MException('GAsolver:Linear','Weight must be in (0,1) non-inclusive.'));
           throw(err);
        end
        bounds = evalin('caller','bounds');
    end
    
    % parents
    p1 = pop(parentIndices(1,:),:);
    p2 = pop(parentIndices(2,:),:);

    % potential offspring
    potential = zeros(numPairs,chromLen,3);
    potential(:,:,1) = weight * p1 + (1 - weight) * p2;
    potential(:,:,2) = (1 + weight) * p1 - weight * p2;
    potential(:,:,3) = (1 + weight) * p2 - weight * p1;
    
    % eliminate out of bounds offspring
%     upper = repmat(bounds(:,1)',numPairs,1);
%     lower = repmat(bounds(:,2)',numPairs,1);
%     good3 = all((potential(:,:,2) <= upper) && (potential(:,:,2) >= lower),2);
%     good3 = all((potential(:,:,3) <= upper) && (potential(:,:,3) >= lower),2);
%     bothbad = not(good2 || good3);
%     % keep 2 copies of inside if both outside offspring are out of bounds;
%     % if only one outside offspring is out of bounds, keep the other 
%     % plus the inside offspring.
%     offspring = [ potential(bothbad,:,1); potential(bothbad,:,1); potential(not(good3),:,2); potential(not(good2),:,3) ];
%     if size(offspring,1) > 0
%         fitness = ones(1,size(offspring,1)) * -inf;
%     else
%         fitness = [];
%     end
    
    % for remaining triplets of offspring, keep best 2 fitness scores
    % evaluate fitness
    objp = evalin('caller', 'p.Results.objParams');
    allFit = ones(3,numPairs) * inf;
    for i=1:3
        allFit(i,:) = obj_func(potential(:,:,i),bounds,objp);
    end
    % sort each triplet of potentials by fitness
    [~,fitIndex] = sort(allFit);
    offspring = [];
    fitness = [];
    % for each set of potentials, grab the ones to keep
    for i=1:3
        individuals = or((fitIndex(2,:)==i),(fitIndex(3,:)==i));
        offspring = [ offspring ; potential(individuals',:,i)];
        fitness = [ fitness , allFit(i,individuals) ];
    end
    
    offspring = offspring(1:numOffspring,:);
    fitness = fitness(1:numOffspring);
    dob = gen * ones(1,numOffspring);
    mob = ones(1,numOffspring);

end

