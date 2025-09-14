%  Create 10 (input) data points evenly spread out in the range [0,π]
p = linspace(0, pi, 10);
% Create the target values for these points
t = sin(p) .* sin(5*p);
plot(p, t, 'o');
net.trainParam.epochs = 1000;
net.trainParam.lr = 0.15;


% net = newfit(p, t, [6], {'tansig' 'purelin'}, 'traingd', '', 'mse', {}, {}, '')
% Here, we don't want classification
% We basically aim for the function remain as it is; 
% thus purelin which is y=x
% So, tansig ranges from [-1,1]. sin also ranges from [-1,1- and sin.sin
% will also be within this range. And we want this to remain as it is.
% net = init(net);
% [trained_net, stats] = train(net, p, t);
% It's overfitting when we put 8 hidden nodes: (n-1) total hidden nodes.
% There is only 1 layer. If we did [8,10] this would means we've 2 layers; 
% 8 nodes for first layer and 10 for the 2nd layer.

% Set Rprop-specific parameters (try tweaking them)
net.trainParam.delta0 = 0.07;   % Initial step size
net.trainParam.deltamax = 40;   % Maximum step size
net.trainParam.delt_inc = 1.2;  % Increase factor
net.trainParam.delt_dec = 0.5;  % Decrease factor
net = newfit(p, t, [6], {'tansig' 'purelin'}, 'trainrp', '', 'mse', {}, {}, '')
net = init(net);
[trained_net, stats] = train(net, p, t);