function [] = plot_colors(som)

x_max = som.layers{1}.dimensions(1);
y_max = som.layers{1}.dimensions(2);

figure

for x=1:x_max,
    for y=1:y_max,
        c = som.iw{1}(x + (y-1) * x_max,:);
        rectangle('Position', [x, y, 1, 1], 'FaceColor', max(min(c, 1), 0));
    end
end
