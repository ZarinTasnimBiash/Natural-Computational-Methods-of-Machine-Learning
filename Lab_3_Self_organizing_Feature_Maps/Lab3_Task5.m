load unknown_data

som = newsom(unknown_data, [10 10], 'hextop', 'linkdist', 1000, 10);
som.trainParam.epochs = 2000; 
% normWineInputs = mapminmax(wineInputs)
[trained_som, stats] = train(som, unknown_data);

figure;
plotsomhits(trained_som, unknown_data(:,1:end));   

figure;
plotsomhits(trained_som, point1);
figure;
plotsomhits(trained_som, point2);
figure;
plotsomhits(trained_som, point3);
figure;
plotsomhits(trained_som, point4);
figure;
plotsomhits(trained_som, point5);

