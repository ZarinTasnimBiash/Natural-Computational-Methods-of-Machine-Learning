function [ nextpop, nextfit, nextdob, nextmob ] = ...
    ga_replace_random( pop, popfitness, popdob, popmob, poprank, ...
    offspring, offspringfitness, offspringDob, offspringMob, ...
    parentIndices, gen, param)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    numOffspring = size(offspring, 1);
    % Randomly rank old population
    
    if param.elitist
        survivors = [size(pop,1) randperm(size(pop,1) - 1)]
    else
        survivors = randperm(size(pop,1));
    end

    oldpop = pop(survivors(end-numOffspring+1:end),:);
    oldfit = popfitness(survivors(end-numOffspring+1:end));
    olddob = popdob(survivors(end-numOffspring+1:end));
    oldmob = popmob(survivors(end-numOffspring+1:end));
    
    if(param.comparative)
        if(strcmp(param.direction,'max'))
            [bestFit, bestFitInd] = max([oldfit; offspringfitness]);
        else
            [bestFit, bestFitInd] = min([oldfit; offspringfitness]);
        end
        offspringInd = logical(bestFitInd - 1);
    else
        offspringInd = true(1,numOffspring);
    end
    
    % New population:
    %   first (pop - offspring) individuals from old pop (may be empty)
    %   offspring selected to survivie (may be all)
    %   individuals form end of old pop to replace unselected offspring
    nextpop = [pop(poprank(survivors(1:end - numOffspring)),:); offspring(offspringInd,:); oldpop(~offspringInd,:)];
    nextfit = [popfitness(poprank(survivors(1:end - numOffspring))) offspringfitness(offspringInd) oldfit(offspringInd)];
    nextdob = [popdob(poprank(survivors(1:end - numOffspring))) offspringDob(offspringInd) olddob(offspringInd)];
    nextmob = [popmob(poprank(survivors(1:end - numOffspring))) offspringMob(offspringInd) oldmob(offspringInd)];
end

