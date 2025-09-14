function [ pop, fitnessVals, dob, mob ] = ga_mutate_inorder( pop, fitnessVals, dob, mob, mutProb, gen, geneType, param)
%GA_MUTATE_UNIFORM Random genes mutated in restricted range
%   Each iteration a start and stop position are selected, and genes within
%   that range are randomly selected for mutation
    
    persistent localParam;
    if gen == 0
        localParam = ga_param_defaults(param, 'step', .1);
    end
    [popSize, chromLen] = size(pop);
    low = randi(chromLen - 1);
    high = randi(chromLen - low) + low;
    mutmask = false(popSize,chromLen);

    if isscalar(mutProb)
        mutmask(:,low:high) = rand(popSize,high-low+1)<mutProb;
    else
        mutmask(:,low:high) = rand(popSize,high-low+1)<mutProb(:,low:high);
    end

    fitnessVals(any(mutmask,2)) = -inf;
    dob(any(mutmask,2)) = gen;
    mob(any(mutmask,2)) = 2;

    % perturbances over a normal distribution:
    step = randn(popSize, chromLen).*localParam.step;
    % perturb each allele if mask indicates to mutate it:
    pop = pop + step .* mutmask;
end

