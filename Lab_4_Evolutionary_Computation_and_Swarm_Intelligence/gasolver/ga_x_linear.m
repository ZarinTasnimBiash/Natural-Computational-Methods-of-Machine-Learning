function [ offspring, fitness, dob, mob ] =...
    ga_x_linear( pop, parentIndices, xmasks, fitnessVals, parentDob, parentMob, numOffspring, gen, param )
%GA_X_LINEAR Linear combination of two sets of parents
%   Detailed explanation goes here
    
    % Determine which parent pairs to leave uncrossed
    % Pairs where both parents are the same individual marked also
    reprodIndices=or(   rand(numOffspring,1)<1-param.prob,...
                        parentIndices(1,:)' == parentIndices(2,:)');
    xmasks(reprodIndices,:)=0;
    
    % save fitness and DOB for preserved parents:
    fitness = ones(1,numOffspring) * -inf;
    fitness(reprodIndices) = fitnessVals(parentIndices(1,reprodIndices));
    dob = gen * ones(1,numOffspring);
    dob(reprodIndices) = parentDob(parentIndices(1,reprodIndices));
    mob = ones(1,numOffspring);
    mob(reprodIndices) = parentMob(parentIndices(1,reprodIndices));
    
    % do the crossover
    offspring = pop(parentIndices(1,:),:) .* (1 - xmasks) +...
                pop(parentIndices(2,:),:) .* xmasks;
end

