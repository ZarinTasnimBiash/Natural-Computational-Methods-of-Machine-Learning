load wine_dataset
som_w = newsom(wineInputs, [5 5], 'hextop', 'linkdist', 1000, 10);
som_w.trainParam.epochs = 2000; 
normWineInputs = mapminmax(wineInputs)

[trained_wine, stats] = train(som_w, wineInputs);

figure;
plotsomhits(trained_wine, wineInputs(:,1:59));   % Winning nodes for F1.

figure;
plotsomhits(trained_wine, wineInputs(:,60:130));   % Winning nodes for F1.

figure;
plotsomhits(trained_wine, wineInputs(:,131:178));   % Winning nodes for F1.

