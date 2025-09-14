load sphere_data.mat;
som1 = newsom(P10, [10 10], 'hextop', 'linkdist', 100, 5);
[som_P10, stats] = train(som1, P10);

som2 = newsom(P20, [10 10], 'hextop', 'linkdist', 100, 5);
[som_P20, stats] = train(som2, P20);

som3 = newsom(P30, [10 10], 'hextop', 'linkdist', 100, 5);
[som_P30, stats] = train(som3, P30);

% figure;
% plotsomhits(som_P10, P10(:,1:100));   % Winning nodes for F1.
% 
% figure;
% plotsomhits(som_P10, P10(:,101:200)); % Winning nodes for F2.

% % Plot the nodes of som_P10 in the input space.
% plotsom(som_P10.iw{1,1}, som_P10.layers{1}.distances)
% hold on
% % Plot the data points in P10.
% plot3(P10(1,:), P10(2,:), P10(3,:), '+k')

% figure;
% plotsomhits(som_P20, P20(:,1:100));   % Winning nodes for F1.
% % 
% figure;
% plotsomhits(som_P20, P20(:,101:200)); % Winning nodes for F2.
% 
% % % Plot the nodes of som_P20 in the input space.
% figure;
% plotsom(som_P20.iw{1,1}, som_P20.layers{1}.distances)
% hold on
% % % Plot the data points in P20.
% plot3(P20(2,:), P20(2,:), P20(3,:), '+k')

% figure;
% plotsomhits(som_P30, P30(:,1:100));   % Winning nodes for F1.
% % 
% figure;
% plotsomhits(som_P30, P30(:,101:200)); % Winning nodes for F2.
% 
% % % Plot the nodes of som_P30 in the input space.
% figure;
% plotsom(som_P30.iw{1,1}, som_P30.layers{1}.distances)
% hold on
% % % Plot the data points in P30.
% plot3(P30(2,:), P30(2,:), P30(3,:), '+k')

figure;
plotsomhits(som_P10, P30(:,1:100)); 
figure;
plotsomhits(som_P10, P30(:,101:200));

figure;
plotsomhits(som_P30, P10(:,1:100)); 
figure;
plotsomhits(som_P0, P10(:,101:200));