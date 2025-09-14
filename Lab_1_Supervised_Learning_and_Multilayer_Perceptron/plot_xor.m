function [] = plot_xor(xor_net)
%plot_xor Plot the output of a 2-2-1 FFNN.
%   Plots the output and the activations of the hidden nodes in
%   a 2-2-1 FFNN, in the domain [-0.5 - 1.5]^2.

res = 50;
range = linspace(-0.5, 1.5, res);
[I1,I2] = meshgrid(range);
    
% Evaluate the first hidden node (H1).
S = xor_net.iw{1}(1,1) * I1 + xor_net.iw{1}(1,2) * I2 + xor_net.b{1}(1);
H1 = feval(xor_net.layers{1}.transferFcn, S);

% Evaluate the second hidden node (H2).
S = xor_net.iw{1}(2,1) * I1 + xor_net.iw{1}(2,2) * I2 + xor_net.b{1}(2);
H2 = feval(xor_net.layers{1}.transferFcn, S);

% Evaluate the output node (O).
S = xor_net.lw{2}(1,1) * H1 + xor_net.lw{2}(1,2) * H2 + xor_net.b{2};
O = feval(xor_net.layers{2}.transferFcn, S);


figure('Name', '2-2-1 network activations', 'Position', [200, 200, 1000, 400]);

% Plot hidden node 1.
subplot(1, 3, 1);
contourf(I1, I2, H1, 15, 'LineStyle', 'none');
hold on;
plot(0, 0, 'wo', 'LineWidth', 2);
plot(0, 1, 'wx', 'LineWidth', 2);
plot(1, 0, 'wx', 'LineWidth', 2);
plot(1, 1, 'wo', 'LineWidth', 2);
title('Activation of hidden node 1');
xlabel('Input 1');
ylabel('Input 2');
colorbar('SouthOutside');
axis square;

% Plot hidden node 2.
subplot(1, 3, 2);
contourf(I1, I2, H2, 15, 'LineStyle', 'none');
hold on;
plot(0, 0, 'wo', 'LineWidth', 2);
plot(0, 1, 'wx', 'LineWidth', 2);
plot(1, 0, 'wx', 'LineWidth', 2);
plot(1, 1, 'wo', 'LineWidth', 2);
title('Activation of hidden node 2');
xlabel('Input 1');
ylabel('Input 2');
colorbar('SouthOutside');
axis square;

% Plot output node.
subplot(1, 3, 3);
contourf(I1, I2, O, 15, 'LineStyle', 'none');
hold on;
plot(0, 0, 'wo', 'LineWidth', 2);
plot(0, 1, 'wx', 'LineWidth', 2);
plot(1, 0, 'wx', 'LineWidth', 2);
plot(1, 1, 'wo', 'LineWidth', 2);
title('Actvivation of output node');
xlabel('Input 1');
ylabel('Input 2');
colorbar('SouthOutside');
axis square;

end


