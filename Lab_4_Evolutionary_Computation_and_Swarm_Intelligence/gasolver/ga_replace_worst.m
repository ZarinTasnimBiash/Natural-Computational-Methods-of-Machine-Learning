function [ nextpop, nextfit, nextdob, nextmob ] = ga_replace_worst( pop, popfitness, popdob, popmob, poprank, offspring, offspringfitness, offspringDob, offspringMob, parentIndices, gen, param)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    numOffspring = size(offspring, 1);
    % New population is survivors (sorted by increasing fitness), 
    % followed by the two complementary sets of offspring.
    % (Fitness gets sorted, too.)
    
    % the worst n current individuals (also first in current sorting)
    worstpop = pop(poprank(1:numOffspring),:);
    worstfit = popfitness(poprank(1:numOffspring));
    
    if(param.comparative)
        if(strcmp(param.direction,'max'))
            [bestFit, bestFitInd] = max([worstfit; offspringfitness]);
        else
            [bestFit, bestFitInd] = min([worstfit; offspringfitness]);
        end
        offspringInd = logical(bestFitInd - 1);
    else
        offspringInd = true(1,numOffspring);
    end
    
    % next population in three segments:
    %   surviving portion of old population (may be empty)
    %   offspring selected for next generation
    %   old pop taking places of unslected offspring (may be empty)
    nextpop = [pop(poprank(numOffspring+1:end),:); offspring(offspringInd,:); worstpop(~offspringInd,:)];
    nextfit = [popfitness(poprank(numOffspring+1:end)) offspringfitness(offspringInd) worstfit(~offspringInd)];
    nextdob = [popdob(poprank(numOffspring+1:end)) offspringDob(offspringInd) popdob(~offspringInd)];
    nextmob = [popmob(poprank(numOffspring+1:end)) offspringMob(offspringInd) popmob(~offspringInd)];
    
    % for elitist strategy, replace last in new pop with best of old
    if param.elitist && numOffspring == size(nextpop,1)
        nextpop(1,:) = pop(poprank(end),:);
        nextfit(1) = popfitness(poprank(end));
        nextdob(1) = popdob(poprank(end));
        nextmob(1) = popmob(poprank(end));
    end
end

