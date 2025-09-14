load iris_dataset
som_flower = newsom(irisInputs, [10 10], 'hextop', 'linkdist', 1000, 10);
som_flower.trainParam.epochs = 2000; 
% normWineInputs = mapminmax(wineInputs)

[trained_flowers, stats] = train(som_flower, irisInputs);

figure;
plotsomhits(trained_flowers, irisInputs(:,1:50));   

figure;
plotsomhits(trained_flowers, irisInputs(:,51:100));   

figure;
plotsomhits(trained_flowers, irisInputs(:,101:150));   
