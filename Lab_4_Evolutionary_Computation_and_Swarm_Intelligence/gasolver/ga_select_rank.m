function [ parentIndices ] = ga_select_rank( popSize, numOffspring, fitnessVals, varargin )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    p = inputParser;
    p.addRequired('popSize',@(x)isnumeric(x)&&isscalar(x)&&x>0);
    p.addRequired('numOffspring',@(x)isnumeric(x)&&isscalar(x)&&x>0);
    p.addRequired('fitnessVals',@(x)isnumeric(x));
    p.addOptional('rank', [], @(x)isnumeric(x));
    p.addOptional('eliteFitness', [], @(x)isnumeric(x));
    p.addOptional('optType','max',@(x)any(strcmp(x,{'max','min'})));
    p.addParamValue('pressure',2,@(x)isnumeric(x)&&isscalar(x)&&x>=1&&x<=2);
    p.addParamValue('sampling','roulette',@(x)any(strcmp(x,{'roulette','stochastic'})));
    p.StructExpand = true;
    p.KeepUnmatched = true;
    p.parse(popSize, numOffspring, fitnessVals, varargin{:});
    
    if isempty(p.Results.rank)
        switch p.Results.optType
            case 'max'
                [sortedFitness rank] = sort(p.Results.fitnessVals, 'ascend');
            case 'min'
                [sortedFitness rank] = sort(p.Results.fitnessVals, 'descend');
        end
    else
        rank = p.Results.rank;
    end
    % section 8.5.5, equation (8.11) seems to be backwards.
    % It says the 'best' individual should have rank 0, but
    % equation (8.11) assigns the lowest fitness to rank 0.
    rankedFitnessVals = (2 - p.Results.pressure + ((0:p.Results.popSize-1)/(p.Results.popSize - 1))*(2 * p.Results.pressure - 2))/ p.Results.popSize;
    cumRankedFitnessVals = cumsum(rankedFitnessVals);
    switch p.Results.sampling
        case 'stochastic'
            markers=rand(1,1)+[1:p.Results.numOffspring]/p.Results.popSize;
            markers(markers>1)=markers(markers>1)-1;
        case 'roulette'
            markers=rand(1,p.Results.numOffspring);
    end
    [temp parentRanks]=histc(markers,[0 cumRankedFitnessVals]);
    parentRanks=parentRanks(randperm(p.Results.numOffspring));
    parentIndices=rank(parentRanks);
end

