p = [[0; 0] [0; 1] [1; 0] [1; 1]];
t = [0 1 1 0];
net = newff(p, t, [2], {'tansig' 'logsig'}, 'traingd', ...
'', 'mse', {}, {}, '');
%net = init(net); [trained_net, stats] = train(net, p, t);
%plot_xor(trained_net)

net.trainParam.epochs = 10000;
net.trainParam.lr = 0.15;
net = init(net); % Reinitialize weights
[trained_net, stats] = train(net, p, t);
plot_xor(trained_net)

sim(trained_net, [0; 0])
sim(trained_net, [0; 1])
sim(trained_net, [1; 0])
sim(trained_net, [1; 1])

figure % Create a new figure.
ax = axes % Get a handle to the figure's axes
hold on % Set the figure to not overwrite old plots.
grid on % Turn on the grid.

% Define learning rates
learning_rates = [0.1, 2, 20];

% Number of training sessions per figure
num_sessions = 10;
num_epochs = 10000; % Keeping a reasonable number of epochs

% Loop over learning rates to create three figures
for i = 1:length(learning_rates)
    lr = learning_rates(i);
    figure; % Create new figure
    ax = axes; % Get figure axes
    hold on; % Allow multiple lines on the same figure
    title(ax, sprintf('lr = %.1f', lr)); % Set title
    
    % Train network 10 times for the same learning rate
    for j = 1:num_sessions
        % Create and configure the network
        net = newff(p, t, [2], {'tansig' 'logsig'}, 'traingd', ...
'', 'mse', {}, {}, '');
        net.trainParam.lr = lr; % Set learning rate
        net.trainParam.epochs = num_epochs;
        
        % Train the network
        net = init(net);
        [trained_net, stats] = train(net, p, t);
        
        % Plot performance curve
        plot(ax, stats.perf);
    end
    
    xlabel('Epochs');
    ylabel('Performance (MSE)');
    grid on;
    legend(arrayfun(@(x) sprintf('Run %d', x), 1:num_sessions, 'UniformOutput', false));
end

