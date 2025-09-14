load("housing.mat");

disp(size(houseInputs));   
disp(size(houseTargets));

% Display statistics of each feature
min_vals = min(houseInputs, [], 2);
max_vals = max(houseInputs, [], 2);
disp([min_vals max_vals]);

normHouseInputs = mapminmax(houseInputs);

min_vals = min(normHouseInputs, [], 2);
max_vals = max(normHouseInputs, [], 2);
disp([min_vals max_vals]);

for run = 1:10
        
    net = newff(normHouseInputs, houseTargets, [20], {'tansig' 'purelin'}, 'trainrp', '', ...
    'mse', {}, {}, 'dividerand');
    
    net.trainParam.max_fail = 1000;
    net.trainParam.epochs = 5000;  % Increase epochs for better learning
    net.trainParam.lr = 0.01;
    net.trainParam.min_grad = 0;
    % net.trainParam.showWindow = false;
    
    net = init(net);
    [trained_net, stats] = train(net, normHouseInputs, houseTargets);
    %early stopping works when you have validation
    %Validation: part of training where you know the error but no backprop
    %on it
    
    % Get predictions
    outputs = sim(trained_net, normHouseInputs);
    
    % Converting one-hot encoded outputs to class labels
    [~, predicted_labels] = max(outputs, [], 1);
    [~, actual_labels] = max(houseTargets, [], 1);
    
    rmse_errors(run) = mean((outputs-houseTargets).^2);

end
disp(rmse_errors);

% Test error is always greater than training error. 

% Validation error is initially higher than test error because it is still in the training process. 
% Test error increases with more epochs because model starts to overfit. 
% Test error is higher than Training error.  