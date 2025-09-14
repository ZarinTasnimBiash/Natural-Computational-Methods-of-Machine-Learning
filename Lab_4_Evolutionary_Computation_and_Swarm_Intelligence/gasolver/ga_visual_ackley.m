function [ ] = ga_visual_ackley( bounds, gen, pop, fit,  elite, efit, params, objparam, h)
%VisFun_Ackley Plot population positions for Ackley's function

% Calculate function surface only once, unless parameters have changed:
persistent X Y Z plotbounds inter
[popsize, dim] = size(pop);
if isempty(X) || (any(strcmp('bounds',fields(params))) && any(params.bounds ~= plotbounds)) || (any(strcmp('interval',fields(params))) && params.interval ~= inter)
    if any(strcmp('bounds',fields(params)))
       plotbounds = params.bounds;
    elseif ~isempty(bounds)
       plotbounds = bounds(1,:);
    else
       plotbounds = [-32,32];
    end
    if any(strcmp('interval',fields(params)))
        inter = params.interval;
    else
        inter = .75;
    end
    [X,Y] = meshgrid(plotbounds(1,1):inter:plotbounds(1,2), plotbounds(1,1):inter:plotbounds(1,2));
    gridpop = [ reshape(X,size(X,1)*size(X,2),1)  reshape(Y,size(Y,1)*size(Y,2),1) ];
    Z = reshape(ga_fit_ackley(gridpop,[],[]), size(X,1), size(X,2));
end

if any(strcmp('type',fields(params)))
    plotfunstr = params.type;
else
    plotfunstr = 'contour';
end
plotfun = str2func(plotfunstr);
if isempty(h)
    h = figure;
end
figure(h);
 plotfun(X,Y,Z);
 hold on
if ~isempty(efit)
    title(['Generation: ' int2str(gen)  ' Fitness: ' num2str(efit(1))]);
end
xlabel('Dimension 1');
ylabel('Dimension 2');
%population:
if ~isempty(pop)
    switch plotfunstr
        case {'contour', 'contourf'}
            popH = scatter(pop(:,1),pop(:,2),'o', 'MarkerEdgeColor', 'none', 'MarkerFaceColor', 'black');
        otherwise
            popH = scatter3(pop(:,1),pop(:,2),fit,'o', 'MarkerEdgeColor', 'none', 'MarkerFaceColor', 'black');
    end
end
hold off
xlim([plotbounds(1,1:2)]);
ylim([plotbounds(1,1:2)]);
drawnow