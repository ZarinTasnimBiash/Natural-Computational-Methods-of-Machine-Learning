function [ ] = ga_plot_fithist( stats )
%GA_PLOT_FITHIST Fitness values over generations

figure();
hold off
plot([0:stats.generations], stats.fitness(1,:),'g-',[0:stats.generations],stats.fitness(2,:),'b-',[0:stats.generations],stats.fitness(3,:),'r-');
hold on
plot([0:stats.generations], stats.fitness(2,:)+stats.fitness(4,:),'c-.',[0:stats.generations],stats.fitness(2,:)-stats.fitness(4,:),'c-.');
%errorbar(stats.fitness(2,:), stats.fitness(4,:));
legend('Best','Average','Worst','Std Dev','Location','NorthWestOutside');
title('Generational Fitness');
xlabel('Generation');
ylabel('Fitness');
xlim([0 stats.generations]);
drawnow