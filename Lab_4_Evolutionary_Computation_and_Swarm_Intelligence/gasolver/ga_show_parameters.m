function [  ] = ga_show_parameters( param )
%GA_SHOW_PARAMETERS Summary of this function goes here
%   Detailed explanation goes here
    f=fields(param);
    for i=1:size(f)
        if isstruct(param.(f{i}))
            display([ f{i} ': ']);
            display(param.(f{i}));
        elseif isnumeric(param.(f{i}))
            if ~isscalar(param.(f{i}))
                display([ f{i} ': ']);
                display(param.(f{i}));
            else
                display([ f{i} ': ' num2str(param.(f{i})) ]);
            end
        elseif islogical(param.(f{i}))
            display([ f{i} ': ' num2str(param.(f{i})) ]);
        else
            display([ f{i} ': ' param.(f{i})]);
            %display(param.(f{i}));
        end
    end

end

