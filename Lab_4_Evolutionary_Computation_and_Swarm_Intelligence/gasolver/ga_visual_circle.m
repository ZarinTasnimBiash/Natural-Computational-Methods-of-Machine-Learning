function [ ] = ga_visual_circle( bounds, gen, population, fitness, eliteIndiv, eliteFitness, param, objparam, h )
%circleVisualizer Plots progress of circle problem for GAsolver
%   Stars are shown with a blue 'x',
%   the current population's locations are marked as small green circles,
%   and the objective value of the best individual is indicated
%   with a large cirlce.
NOP = 100;
figure(h);

% plot stars:
scatter(objparam.star(:,1),objparam.star(:,2),'x');
hold on

%population:
popH = scatter(population(:,1),population(:,2),'o');

% best circle found up to now:
THETA=linspace(0,2*pi,NOP);
RHO=ones(1,NOP)*eliteFitness(1);
[circX,circY] = pol2cart(THETA,RHO);
circX=circX+eliteIndiv(1,1);
circY=circY+eliteIndiv(1,2);
circH = plot(circX,circY,'b-');

title(['X: ' num2str(eliteIndiv(1,1)) '; Y: ' num2str(eliteIndiv(1,2)) '; Radius: ' num2str(eliteFitness(1))]);
xlabel('X');
ylabel('Y');
hold off
axis square;
drawnow;
end
