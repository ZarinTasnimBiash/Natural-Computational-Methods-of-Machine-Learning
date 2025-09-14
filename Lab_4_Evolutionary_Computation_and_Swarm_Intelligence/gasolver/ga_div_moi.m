function [ diversity ] = ga_div_moi( pop, fit )
%GA_DIVERSITY_MOI Moment of Inertia
%
%   Moment of inertia as a measure of population diversity, a relatively
%   efficient phenotypic diversity measure described in "Measurement of
%   Population Diversity", Morrison and De Jong, in Artificial Evolution,
%   Springer, 2001.

centroid = repmat(sum(pop)/size(pop,1),size(pop,1),1);
diff = pop - centroid;
diversity = sum(sum(diff .* diff));
end

