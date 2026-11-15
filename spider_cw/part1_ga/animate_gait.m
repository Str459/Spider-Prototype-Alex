function animate_gait(detail, geom, fcfg, outDir, varargin)
%ANIMATE_GAIT 3D renderings of the evolved gait.
%   pose_snapshots.png  six moments of one cycle
%   gait.gif            the spider walking for nCycles cycles. The body is
%                       moved forward by the distance the planted feet push
%                       it, so a foot that is truly planted stays still on
%                       the ground (its footprints line up) while the body
%                       passes over it. Sliding footprints = slip.
%   Options: 'nCycles' (2), 'frameStep' (5, i.e. 60 GIF frames per cycle),
%            'gif' (true).

p = inputParser;
addParameter(p, 'nCycles', 2);
addParameter(p, 'frameStep', 5);
addParameter(p, 'gif', true);
parse(p, varargin{:});

F = size(detail.angles, 2);
A = reshape(detail.angles, F, 24);
stance = reshape(detail.stance, F, 8);
feet = reshape(detail.feet, F, 8, 3);
vx = reshape(detail.vx, F, 1);
perCycle = sum(vx);
bodyX = [0; cumsum(vx(1:end - 1))];   % body position at each frame of cycle 1

% Snapshots -----------------------------------------------------------------
fig = new_figure(1300, 750);
snaps = round(linspace(1, F, 7));
snaps = snaps(1:6);
for k = 1:6
    f = snaps(k);
    ax = subplot(2, 3, k);
    setup_axes(ax, 0, fcfg.groundZ);
    draw_spider(ax, A(f, :), geom, 'planted', stance(f, :), 'labels', k == 1);
    title(ax, sprintf('Frame %d of %d', f, F));
end
sgtitle('Evolved gait: snapshots through one cycle (green = planted foot)');
exportgraphics(fig, fullfile(outDir, 'pose_snapshots.png'), 'Resolution', 110);
close(fig);

if ~p.Results.gif
    return
end

% Animation -------------------------------------------------------------------
gifFile = fullfile(outDir, 'gait.gif');
fig = new_figure(640, 520);
ax = axes(fig);
prints = zeros(0, 3);
map = [];
first = true;
for cyc = 0:p.Results.nCycles - 1
    for f = 1:p.Results.frameStep:F
        bx = cyc * perCycle + bodyX(f);
        cla(ax);
        setup_axes(ax, bx, fcfg.groundZ);
        world = squeeze(feet(f, :, :)) + [bx 0 0];
        prints = [prints; world(stance(f, :), :)]; %#ok<AGROW>
        plot3(ax, prints(:, 1), prints(:, 2), prints(:, 3), '.', 'Color', [0.55 0.55 0.55], 'MarkerSize', 4);
        draw_spider(ax, A(f, :), geom, 'offset', [bx 0 0], 'planted', stance(f, :), 'labels', false);
        title(ax, sprintf('Cycle %d, frame %d / %d   distance %.2f', cyc + 1, f, F, bx));
        img = print(fig, '-RGBImage', '-r80');
        if first
            [ind, map] = rgb2ind(img, 128);
            imwrite(ind, map, gifFile, 'gif', 'LoopCount', Inf, 'DelayTime', 0.05);
            first = false;
        else
            ind = rgb2ind(img, map);
            imwrite(ind, map, gifFile, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
        end
    end
end
close(fig);
end

function setup_axes(ax, bx, groundZ)
% Ground grid fixed in the world, camera following the body.
hold(ax, 'on');
xl = [bx - 4.5, bx + 4.5];
gx = floor(xl(1)):ceil(xl(2));
fill3(ax, [xl(1) xl(2) xl(2) xl(1)], [-4 -4 4 4], groundZ * [1 1 1 1], ...
    [0.93 0.95 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
for g = gx
    plot3(ax, [g g], [-4 4], [groundZ groundZ], '-', 'Color', [0.8 0.8 0.8]);
end
for g = -4:4
    plot3(ax, xl, [g g], [groundZ groundZ], '-', 'Color', [0.8 0.8 0.8]);
end
axis(ax, 'equal');
xlim(ax, xl); ylim(ax, [-4 4]); zlim(ax, [groundZ - 0.2, 1.5]);
view(ax, 35, 25);
grid(ax, 'on');
xlabel(ax, 'X (forward)'); ylabel(ax, 'Y'); zlabel(ax, 'Z');
end
