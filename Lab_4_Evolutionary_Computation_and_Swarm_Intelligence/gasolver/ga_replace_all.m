function [ nextpop, nextfit, nextdob, nextmob ] = ga_replace_all( pop, popfitness, popdob, popmob, poprank, offspring, offspringfitness, offspringDob, offspringMob, parentIndices, gen, param)
%GA_REPLACE_ALL Replace old population with new population
%   This really only makes sense for GGA.
%   If there aren't enough offspring to replace the population,
%   punt to replace_random and replace as many as possible.
    
    if (size(offspring,1) ~= size(pop,1))
        % can't replace all with fewer offspring!
        % choose randomly, instead
        [ nextpop, nextfit, nextdob, nextmob ] = ...
            ga_replace_random( pop, popfitness, popdob, popmob, poprank, ...
            offspring, offspringfitness, offspringDob, offspringMob, ...
            parentIndices, gen, param);
    else

        if(param.comparative)
            if(strcmp(param.direction,'max'))
                [bestFit, bestFitInd] = max([popfitness; offspringfitness]);
            else
                [bestFit, bestFitInd] = min([popfitness; offspringfitness]);
            end
            offspringInd = logical(bestFitInd - 1);
        else
            offspringInd = true(1,size(offspring,1));
        end
        nextpop = [ pop(~offspringInd,:); offspring(offspringInd,:) ];
        nextfit = [ popfitness(~offspringInd) offspringfitness(offspringInd) ];
        nextdob = [ popdob(~offspringInd) offspringDob(offspringInd) ];
        nextmob = [ popmob(~offspringInd) offspringMob(offspringInd) ];

        if param.elitist
            repIndex = randi(size(nextpop,1));
            nextpop(repIndex,:) = pop(poprank(end),:);
            nextfit(repIndex) = popfitness(poprank(end));
            nextdob(repIndex) = popdob(poprank(end));
            nextmob(repIndex) = popmob(poprank(end));
        end
    end
end

