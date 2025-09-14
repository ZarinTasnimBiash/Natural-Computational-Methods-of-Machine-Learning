function [ term_string, finished ] =...
    ga_stop_generations( ~, ~, ~, ~, ~ )
%GA_STOP_GENERATIONS Stop when max generation count reached.
%   This doesn'y actually do anything.
    term_string = 'maximum generation count reached.';
    finished = false;
end

