function [ pop ] = ga_init_bit( popSize, chromLen, bounds, params, varargin)
%ga_init_bit Initialize binary population randomly. 

    pop=rand(popSize,chromLen)<.5;
end

