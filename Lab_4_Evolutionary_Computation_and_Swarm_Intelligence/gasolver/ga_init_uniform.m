function [ pop ] = ga_init_uniform( popSize, chromLen, bounds, params, varargin )
%ga_init_uniform Initial population with uniformly distributed gene values 
%   Detailed explanation goes here

    pop = rand(popSize,chromLen);
    for i=1:chromLen
        pop(:,i) = (bounds(i,2) - bounds(i,1))*pop(:,i) + bounds(i,1);
    end
end

