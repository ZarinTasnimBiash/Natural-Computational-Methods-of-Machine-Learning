function [ diversity ] = ga_div_colvar( pop, fit )
%GA_DIV_COLVAR Sum of Column Variances
%
%   Commonly used phenotypic measure of population diversity.
%   Claculates the sum of variances of individual genes.

    diversity = sum(var(pop));

end

