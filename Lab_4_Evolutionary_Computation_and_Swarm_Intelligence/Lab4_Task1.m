warning('off','all');
star = GAparams;
% star.objParams.star = star1;
star.objParams.star = star3;
% star.objParams.star = star1;

%fitness does not require the select.pressure
star.select.func = 'rank';
star.select.pressure = 2;

star.visual.active = 1;
star.visual.func = 'circle';
[best, fit, stat] = GAsolver(2, [0 20 ; 0 20], ...
'circle', 50, 100, star);

ga_plot_diversity(stat);
