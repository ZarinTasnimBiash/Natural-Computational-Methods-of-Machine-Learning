function [ term_string, finished ] =...
    ga_stop_generations( gen, fitness, staticGens, elapsed, params )
%GA_STOP_GENERATIONS Stop when max generation count reached.
%   This doesn'y actually do anything.
    term_string = '';
    finished =  staticGens >= params.window;
    if finished  
        term_string = ['No improvement after ' int2str(staticGens) ' generations.'];
    end
end

