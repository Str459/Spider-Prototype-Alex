function plot_gait_analysis(detail, geom, fcfg, outDir)
%PLOT_GAIT_ANALYSIS Plots of one decoded gait (from gait_fitness's detail).
%   joint_angles.png  all 24 joint angles over the cycle, grouped by joint
%   foot_heights.png  each foot's height above the ground, and the gait
%                     diagram: black bar = foot planted. A good walk shows
%                     alternating groups of legs, never everyone lifted.
%   stability.png     stability margin, number of feet down, body speed

[A, stance, feet, margin, nStance, vx] = unpack(detail);
F = size(A, 1);
frames = 1:F;

% Joint angles --------------------------------------------------------------
fig = new_figure(1000, 750);
jointNames = {'coxa (a)', 'femur (b)', 'tibia (c)'};
for j = 1:3
    subplot(3, 1, j);
    plot(frames, rad2deg(A(:, j:3:end)), 'LineWidth', 1.2);
    grid on;
    ylabel([jointNames{j} ' [deg]']);
    if j == 1
        legend(geom.labels, 'Orientation', 'horizontal', 'Location', 'northoutside');
        title('Joint angles over one gait cycle (raw model angles)');
    end
end
xlabel('Frame');
exportgraphics(fig, fullfile(outDir, 'joint_angles.png'), 'Resolution', 110);
close(fig);

% Foot heights and gait diagram ------------------------------------------
fig = new_figure(1000, 700);
subplot(2, 1, 1);
plot(frames, squeeze(feet(:, :, 3)) - fcfg.groundZ, 'LineWidth', 1.2);
hold on;
yline(0, 'k-', 'ground');
yline(fcfg.clearance, 'k:', 'required clearance');
grid on;
ylabel('Foot height above ground');
legend(geom.labels, 'Orientation', 'horizontal', 'Location', 'northoutside');
title('Foot heights');
subplot(2, 1, 2);
imagesc(frames, 1:8, ~stance');
colormap(gca, [0 0 0; 1 1 1]);
yticks(1:8);
yticklabels(geom.labels);
xlabel('Frame');
title('Gait diagram (black = foot planted)');
exportgraphics(fig, fullfile(outDir, 'foot_heights.png'), 'Resolution', 110);
close(fig);

% Stability -----------------------------------------------------------------
fig = new_figure(1000, 700);
subplot(3, 1, 1);
plot(frames, margin, 'LineWidth', 1.5);
hold on;
yline(fcfg.minMargin, 'r--', 'required');
yline(0, 'k-', 'tips over');
grid on;
ylabel('Stability margin');
title('Distance from body centre to edge of support polygon');
subplot(3, 1, 2);
stairs(frames, nStance, 'LineWidth', 1.5);
hold on;
yline(fcfg.minStance, 'r--', 'required');
ylim([0 8.5]);
grid on;
ylabel('Feet down');
subplot(3, 1, 3);
plot(frames, vx * F, 'LineWidth', 1.5);
grid on;
ylabel('Body speed [units/cycle]');
xlabel('Frame');
exportgraphics(fig, fullfile(outDir, 'stability.png'), 'Resolution', 110);
close(fig);
end

function [A, stance, feet, margin, nStance, vx] = unpack(d)
F = size(d.angles, 2);
A = reshape(d.angles, F, 24);
stance = reshape(d.stance, F, 8);
feet = reshape(d.feet, F, 8, 3);
margin = reshape(d.margin, F, 1);
nStance = reshape(d.nStance, F, 1);
vx = reshape(d.vx, F, 1);
end
