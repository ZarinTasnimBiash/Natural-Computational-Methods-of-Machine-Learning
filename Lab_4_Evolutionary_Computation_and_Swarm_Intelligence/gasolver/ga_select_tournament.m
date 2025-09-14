function [ parentIndices ] = ga_select_tournament( popSize, numOffspring, fitnessVals, varargin )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    p = inputParser;
    p.addRequired('popSize',@(x)isnumeric(x)&&isscalar(x)&&x>0);
    p.addRequired('numOffspring',@(x)isnumeric(x)&&isscalar(x)&&x>0);
    p.addRequired('fitnessVals',@(x)isnumeric(x));
    p.addOptional('rank', [], @(x)isnumeric(x));
    p.addOptional('eliteFitness', [], @(x)isnumeric(x));
    p.addOptional('optType','max',@(x)any(strcmp(x,{'max','min'})));
    p.addParamValue('size',2,@(x)isnumeric(x)&&isscalar(x)&&x>=2);
    p.StructExpand = true;
    p.KeepUnmatched = true;
    p.parse(popSize, numOffspring, fitnessVals, varargin{:});
    
    % start with a random set of potential parents
    parentIndices = randi(p.Results.popSize, 1, p.Results.numOffspring);
    for i=1:p.Results.size-1
        % choose a set of competitors
        pool = randi(p.Results.popSize, 1, p.Results.numOffspring);
        differentialFitness = p.Results.fitnessVals(parentIndices) - p.Results.fitnessVals(pool);
        switch p.Results.optType
            case 'max'
                parentIndices(differentialFitness< 0) = pool(differentialFitness < 0);
            case 'min'
                parentIndices(differentialFitness> 0) = pool(differentialFitness > 0);
        end
    end
end

