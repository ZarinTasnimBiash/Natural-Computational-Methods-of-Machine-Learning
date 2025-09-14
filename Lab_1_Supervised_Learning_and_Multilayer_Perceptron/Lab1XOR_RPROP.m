p = [[0; 0] [0; 1] [1; 0] [1; 1]];
t = [0 1 1 0];

net = newff(p, t, [2], {'tansig' 'logsig'}, 'trainrp', ...
'', 'mse', {}, {}, '');

net.trainParam.epochs = 1000;
net.trainParam.lr = 0.15;
net = init(net); % Reinitialize weights
[trained_net, stats] = train(net, p, t);
% plot_xor(trained_net)

sim(trained_net, [0; 0])
sim(trained_net, [0; 1])
sim(trained_net, [1; 0])
sim(trained_net, [1; 1])

% figure % Create a new figure.
% ax = axes % Get a handle to the figure's axes
% hold on % Set the figure to not overwrite old plots.
% grid on % Turn on the grid.

% Number of training sessions per figure
num_sessions = 10;
num_epochs = 100; % Keeping a reasonable number of epochs

% Train the network 10 times and plot performance curves
for i = 1:num_sessions
     % Create and configure network
     net = newff(p, t, [2], {'tansig', 'logsig'}, 'trainrp');
     net.trainParam.epochs = num_epochs;

     % Set Rprop-specific parameters (tweak)
     net.trainParam.delta0 = 0.07;   % Initial step size
     net.trainParam.deltamax = 50;   % Maximum step size
     net.trainParam.delt_inc = 1.2;  % Increase factor
     net.trainParam.delt_dec = 0.5;  % Decrease factor
 
     % Train network
     net = init(net);
     [trained_net, stats] = train(net, p, t);
 
     % Plot performance curve
     plot(stats.perf, 'Color', colors(i,:));
 
 
end
xlabel('Epochs');
ylabel('Performance (MSE)');
grid on;
legend(arrayfun(@(x) sprintf('Run %d', x), 1:num_sessions, 'UniformOutput', false));
 
 % Save the plot
 saveas(gcf, 'rprop_performance.png');