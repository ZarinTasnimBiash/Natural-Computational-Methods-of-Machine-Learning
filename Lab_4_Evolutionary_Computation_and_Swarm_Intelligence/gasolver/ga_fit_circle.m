function [ f ] = ga_fit_circle( population, bounds, param )
%ObjFun_circle largest circle problem for GAsolver
%   Fitness is the radius of the largest circle that can be drawn around
%   each point in population, without enclosing any of the 'stars', as
%   given by the coordinates in param.star

star = param.star;
[popsize, dim] = size(population);
numstars = size(star);
f = [];
for j=1:popsize
    % distance to each star:
    d = [];
    for k=1:numstars
%         d(k) = 0;
%         for b=1:dim
%            d(k) = d(k) + (star(k,b) - population(j,b))^2  
%         end
%         d(k) = sqrt(d(k));
        d(k)=sqrt(sum( (star(k,:) - population(j,:)) .^2 ));
    end
    % distance to each bound:
    for b = 1:dim
        % distance to lower bound
        %d(numstars + b*2 - 1) = abs(population(j,b) - bounds(b,2));
        % distance to upper bound
        %d(numstars + b*2) = abs(bounds(b,1) - population(j,b));
        d = [d  abs(population(j,b) - bounds(b,2)) abs(bounds(b,1) - population(j,b))];
    end
    %d
    % distance to closest obstacle:
    f(j) = min(d);
end
end

