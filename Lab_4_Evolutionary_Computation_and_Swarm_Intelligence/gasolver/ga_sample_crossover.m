function ga_sample_crossover( pop, numOffspring, xfunc, xparams)
%GA_SAMPLE_CROSSOVER Plot offspring from one set of parents
%   Detailed explanation goes here
    if any(strcmp('bounds',fields(xparams)))
        bounds = xparams.bounds;
    end
    
    obj_fun = @ga_fit_random;
    p.Results.objParams = struct();
    select_func = @ga_select_test;
    
    [offspring, fitness, ~, ~] = xfunc(pop, [], rand(1,size(pop,1)), zeros(1,size(pop,1)), zeros(1,size(pop,1)), [], numOffspring,...
        0, select_func, struct(), [], xparams);
    figure();
    scatter(offspring(:,1), offspring(:,2), 'k', 'filled');
    hold on;
    scatter(pop(:,1),pop(:,2),'b','filled');
    for i=1:size(pop,1)
       text(pop(i,1),pop(i,2),['p_' int2str(i)]); 
    end
end

