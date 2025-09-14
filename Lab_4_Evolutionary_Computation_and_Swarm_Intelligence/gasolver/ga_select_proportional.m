function [ parentIndices ] = ga_select_proportional( popSize, numOffspring, fitnessVals, varargin )
%GA_SELECT_PROPORTIONAL Fitness Proportional Selection
%
%   Random selection, with probability of choosing a genome proportional to
%   its normalized fitness value. By default uses roulette sampling, but
%   can optionally use stochastic universal sampling (SUS) instead.

    p = inputParser;
    p.addRequired('popSize',@(x)isnumeric(x)&&isscalar(x)&&x>0);
    p.addRequired('numOffspring',@(x)isnumeric(x)&&isscalar(x)&&x>0);
    p.addRequired('fitnessVals',@(x)isnumeric(x));
    p.addOptional('rank', [], @(x)isnumeric(x));
    p.addOptional('eliteFitness', [], @(x)isnumeric(x));
    p.addOptional('optType','max',@(x)any(strcmp(x,{'max','min'})));
    p.addParamValue('sampling','roulette',@(x)any(strcmp(x,{'roulette','stochastic'})));
    p.StructExpand = true;
    p.KeepUnmatched = true;
    p.parse(popSize, numOffspring, fitnessVals, varargin{:});
    
    % Normalize the fitness values and then create an array with the
    % cumulative normalized fitness values (the last value in this array
    % will be 1)
    % First we scale the fitness values to the range (0,1], based on the
    % best objective value found so far.
    switch p.Results.optType
        case 'max'
            propFitness = 1 ./ (1 + p.Results.eliteFitness(1) - p.Results.fitnessVals);
        case 'min'
            propFitness = 1 ./ (1 + p.Results.fitnessVals - p.Results.eliteFitness(1));
    end
    % These scaled values are normalized so that they sum to 1,
    % and an array of cumulative values is made.
    % This holds the values computed in roulette wheel selection (alg 8.2),
    % it just does it all at once instead of incrementally.
    cumNormFitnessVals=cumsum(propFitness/sum(propFitness));

    % Now we use either Stochastic Universal or Roulette Wheel Sampling
    % to select parents for all crossover operations.
    % The markers array holds a series of random numbers in (0,1]:
    switch p.Results.sampling
        case 'stochastic'
            markers=rand(1,1)+[1:p.Results.numOffspring]/p.Results.popSize;
            markers(markers>1)=markers(markers>1)-1;
        case 'roulette'
            markers=rand(1,p.Results.numOffspring);
    end
    % parentIndices shows the index of the parents that correspond
    % to each of the random values in markers.
    [temp parentIndices]=histc(markers,[0 cumNormFitnessVals]);
    parentIndices=parentIndices(randperm(p.Results.numOffspring)); 
end

