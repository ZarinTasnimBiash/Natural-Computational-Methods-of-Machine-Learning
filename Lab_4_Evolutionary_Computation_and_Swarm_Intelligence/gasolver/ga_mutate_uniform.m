function [ pop, fitnessVals, dob, mob ] = ga_mutate_uniform( pop, fitnessVals, dob, mob, mutProb, gen, geneType, param)
%GA_MUTATE_UNIFORM Random genes mutated
%   Detailed explanation goes here
    
    
    persistent localParam popmax popmin popsize chromlen;
    if gen == 0
        [popsize, chromlen] = size(pop);
        switch geneType
            case 'binary'
                geneTypeID = 0;
            case 'float'   
                geneTypeID = 1;
                bounds = evalin('caller', 'bounds');
                popmax = repmat(bounds(:,1)',popsize,1);
                popmin = repmat(bounds(:,2)',popsize,1);
            otherwise        
                geneTypeID = -1;
        end
        localParam = ga_param_defaults(param, 'geneType', geneTypeID);
    end
    
    mutmask = rand(popsize,chromlen)<mutProb;

    % delete fitness of all individuals about to be mutated
    fitnessVals(any(mutmask,2)) = -inf;
    dob(any(mutmask,2)) = gen;
    mob(any(mutmask,2)) = 2;
    
    switch localParam.geneType
        case 0  %binary
            pop = xor(pop,mutmask);
        case 1  %float
            % each 'flipped' position has equal chance of increasing or
            % decreasing:
            direction = rand(popsize,chromlen)<0.5;
            upmask = and(mutmask,direction);
            downmask = and(mutmask,~direction);
            magnitude = rand(popsize,chromlen);
            pop = pop + magnitude .* upmask .* (popmax - pop);
            pop = pop - magnitude .* downmask .* (pop - popmin);
    end
end

