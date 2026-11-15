function draw_spider(ax, pose, geom, varargin)
%DRAW_SPIDER Draw one 1x24 pose into existing axes.
%   DRAW_SPIDER(ax, pose, geom, 'offset', [x y z], 'planted', mask, 'labels', true)
%   Colours follow the lecturer's plot_spider_pose: coxa black, femur blue,
%   tibia red. Planted feet are filled green, lifted feet are open red.
%   'offset' moves the whole spider (used to show it walking forward).

p = inputParser;
addParameter(p, 'offset', [0 0 0]);
addParameter(p, 'planted', true(1, 8));
addParameter(p, 'labels', true);
parse(p, varargin{:});
o = p.Results.offset;
planted = logical(p.Results.planted);

[~, J] = spider_fk(pose(:)', geom);
J = reshape(J, 8, 4, 3);
hold(ax, 'on');

t = linspace(0, 2 * pi, 60);
fill3(ax, o(1) + geom.bodyA * cos(t), o(2) + geom.bodyB * sin(t), o(3) + zeros(size(t)), ...
    [0.6 0.6 0.6], 'EdgeColor', 'k', 'LineWidth', 1.5);
plot3(ax, o(1) + geom.bodyA + 0.2, o(2), o(3), 'r^', 'MarkerFaceColor', 'r', 'MarkerSize', 7);

segCol = {'k', 'b', 'r'};
for leg = 1:8
    P = squeeze(J(leg, :, :)) + o;          % 4 x 3 joint positions
    for s = 1:3
        plot3(ax, P(s:s + 1, 1), P(s:s + 1, 2), P(s:s + 1, 3), segCol{s}, 'LineWidth', 2);
    end
    if planted(leg)
        plot3(ax, P(4, 1), P(4, 2), P(4, 3), 'o', 'MarkerSize', 6, ...
            'MarkerFaceColor', [0 0.6 0], 'MarkerEdgeColor', [0 0.4 0]);
    else
        plot3(ax, P(4, 1), P(4, 2), P(4, 3), 'o', 'MarkerSize', 6, 'MarkerEdgeColor', 'r');
    end
    if p.Results.labels
        lp = o + geom.basePos(leg, :) + 0.25 * [cos(geom.baseAngle(leg)), sin(geom.baseAngle(leg)), 0];
        text(ax, lp(1), lp(2), lp(3) + 0.1, geom.labels{leg}, 'FontSize', 8, 'FontWeight', 'bold');
    end
end
end
