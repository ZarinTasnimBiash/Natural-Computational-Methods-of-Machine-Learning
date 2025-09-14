function [ ] = ga_plot_diversity( stats )
%GA_PLOT_diversity Fitness values over generations

figure();
plot([0:stats.generations],stats.diversity);
hold on
title('Population Diversity');
xlabel('Generation');
ylabel('Diversity');
xlim([0 stats.generations]);
drawnow