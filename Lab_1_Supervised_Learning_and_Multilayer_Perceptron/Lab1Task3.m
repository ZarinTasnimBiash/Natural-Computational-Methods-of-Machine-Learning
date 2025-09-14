load wine_dataset;
% always set your input for the network to the number of columns, in this case 13

% Display dataset dimensions
% disp(size(wineInputs));   % Should output [13, 178] (13 features, 178 samples)
% disp(size(wineTargets));  % Should output [3, 178] (3-class classification) in one-hot encoding format

% Display statistics of each feature
min_vals = min(wineInputs, [], 2);
max_vals = max(wineInputs, [], 2);
% disp([min_vals max_vals]);

% % Get the number of samples
% numSamples = size(wineInputs, 2);
% 
% % Generate a random permutation of indices
% randIndices = randperm(numSamples);
% 
% % Shuffle inputs and targets
% shuffledInputs = wineInputs(:, randIndices);
% shuffledTargets = wineTargets(:, randIndices);

% Since features have very different scales, we normalize them to [−1,1] using mapminmax
% wineInputs = mapminmax(wineInputs); %-> restricted
wineInputs = mapminmax(wineInputs)

% Defining the network architecture
hiddenNodes = 7; 
net = newff(wineInputs, wineTargets, [hiddenNodes], {'tansig', 'softmax'}, 'traingd','', 'mse', {}, {}, '');

% Set training parameters
net.trainParam.epochs = 1000;  % Increase epochs for better learning
net.trainParam.lr = 0.01;
% net.trainParam.delta0 = 0.07;
% net.trainParam.deltamax = 50;  % Maximum step size for Rprop
% net.trainParam.delt_inc = 1.2; % Increase factor
% net.trainParam.delt_dec = 0.5; % Decrease factor

% Train the network
net = init(net);
[trained_net, stats] = train(net, wineInputs, wineTargets);

% Get predictions
outputs = sim(trained_net, wineInputs);

% Converting one-hot encoded outputs to class labels
[~, predicted_labels] = max(outputs, [], 1);
[~, actual_labels] = max(wineTargets, [], 1);

% Compute accuracy
accuracy = sum(predicted_labels == actual_labels) / length(actual_labels) * 100;
fprintf("Training Accuracy: %.2f%%\n", accuracy);

% Plot confusion matrix
plotconfusion(wineTargets, outputs);