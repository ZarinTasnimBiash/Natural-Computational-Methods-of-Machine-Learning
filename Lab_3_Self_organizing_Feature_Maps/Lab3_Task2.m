load rgb_data
trained_som = newsom(RGB, [10 10], 'gridtop', 'linkdist', 1000, 10);
trained_som.trainParam.epochs = 1000
[trained_som, stats] = train(trained_som, RGB);
plot_colors(trained_som)

