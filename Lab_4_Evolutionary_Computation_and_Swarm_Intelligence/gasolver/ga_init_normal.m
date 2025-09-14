function [ pop ] = ga_init_normal( popSize, chromLen, bounds, params, varargin)
%ga_init_normal Initial population with normally distributed gene values 
%   The peak of the distribution is determined by the 'start' parameter,
%   and can be 'min', 'mean', or 'max' for left, middle, or right norms.
    
    pop = randn(popSize,chromLen) * params.variance;
    switch params.start
        case 'min'
            pop = abs(pop);
            peaks = repmat(bounds(:,1)',popSize,1);
        case 'max'
            pop = abs(pop) * -1;
            peaks = repmat(bounds(:,2)',popSize,1);
        case 'mean'
            peaks = repmat(((bounds(:,2) + bounds(:,1)) / 2)',popSize,1);
    end
    pop = pop + peaks;
end

