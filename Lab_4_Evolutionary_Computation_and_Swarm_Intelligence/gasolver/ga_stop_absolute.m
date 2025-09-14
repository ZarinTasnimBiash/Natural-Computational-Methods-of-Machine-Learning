function [ term_string, finished ] =...
    ga_stop_absolute( ~, fitness, ~, ~, params )
%GA_STOP_GENERATIONS Stop when max generation count reached.
%   This doesn'y actually do anything.
    term_string = '';
    switch params.direction
        case 'max'
            finished = max(fitness) >= params.value - params.error;
        case 'min'
            finished = min(fitness) <= params.value + params.error;
    end
    if finished
        term_string = ['Optimum value ' num2str(params.value) ' +/-' num2str(params.error) ' reached.'];
    end
end

