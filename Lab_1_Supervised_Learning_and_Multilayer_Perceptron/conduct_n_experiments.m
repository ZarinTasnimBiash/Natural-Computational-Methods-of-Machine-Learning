%% Code example showing how to conduct and store N experiments

% Initial setup; e.g.:
p = [[0; 0] [0; 1] [1; 0] [1; 1]];
t = [0 1 1 0];

n_experiments = 42;                % How many experiments to run?
outputs = cell(n_experiments,1);   % Create cell array for storing results
for i = 1:n_experiments
    % vvv Lab instruction code goes here vvv
    % For example:
    net = newff(p, t, (2), {'tansig' 'logsig'}, 'traingd', ...
                '', 'mse', {}, {}, '');
    net = init(net);    % Note: you have to initialize every loop! (Why?)
    [trained_net, stats] = train(net, p, t);
    % ^^^ Lab instruction code goes here ^^^
    
    outputs{i} = trained_net; % Store the trained nets at right-hand side
end

% Analysis code of output here; e.g.:
% plot_xor(outputs{some_particular_experiment});
% ...and other code from lab instructions

% Happy hacking!