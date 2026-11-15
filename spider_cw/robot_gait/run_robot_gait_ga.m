function results = run_robot_gait_ga(seeds, varargin)
%RUN_ROBOT_GAIT_GA Evolve walking gaits for the Webots robot with the Part 1 GA.
%   run_robot_gait_ga            % seeds 0..4
%   run_robot_gait_ga(0:2, 'popSize', 80, 'maxGen', 200)
%
%   Uses the GA engine in this folder (ga_run.m: tournament
%   selection, two-point crossover, adaptive Gaussian mutation, elitism)
%   on the 11 robot gait genes (robot_gait_genes.m), scored by
%   robot_gait_fitness.m. Each seed's best gait is exported as a CANDIDATE.
%   The final gait is then chosen by testing the candidates in Webots
%   physics (select_in_webots.py), not by hand.
%
%   Writes spider_cw/outputs/robot_gait/seed<k>/:
%     joint_table.csv  200 x 24 Webots joint angles [L1 hip thigh knee, L2 ...,
%                      R1 hip thigh knee], one frame per row, one cycle
%     stance.csv       200 x 8, 1 = foot planted (same leg order)
%     stand.csv        1 x 24 standing pose (all feet at stride centre)
%     startup.csv      4 cycles that go from standing to the full gait,
%                      growing the stride one step at a time (+ startup_stance.csv)
%     meta.csv         genes, period, speed and every fitness term
%     fitness.png      GA progress;  gait_check.png  joints vs limits,
%                      joint speeds, clearance, gait diagram, foot paths

if nargin < 1 || isempty(seeds)
    seeds = 0:4;
end
here = fileparts(mfilename('fullpath'));
% (GA engine files are in this folder)

g     = robot_geometry();
genes = robot_gait_genes();
fcfg  = robot_fitness_config();
cfg   = ga_default_config();                      % Part 1 defaults (two-point etc.)
cfg.popSize = 60;
cfg.maxGen = 150;
cfg.stallGenerations = 60;
cfg.printEvery = 25;
for k = 1:2:numel(varargin)
    cfg.(varargin{k}) = varargin{k + 1};
end

problem.nGenes   = genes.n;
problem.lb       = genes.lb;
problem.ub       = genes.ub;
problem.circular = genes.circular;
problem.evaluate = @(pop) robot_gait_fitness(pop, fcfg, g);

outRoot = fullfile(here, '..', 'outputs', 'robot_gait');
results = table();
for sd = seeds
    cfg.seed = sd;
    fprintf('=== Robot gait GA, seed %d ===\n', sd);
    r = ga_run(problem, cfg);
    outDir = fullfile(outRoot, sprintf('seed%d', sd));
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    export_gait(r, outDir, g, fcfg, genes);
    plot_ga_history(r, outDir, sprintf('robot gait seed %d', sd));
    c = r.bestComp;
    results = [results; table(sd, r.bestScore, c.speed, c.maxJointVel, c.maxTorque, ...
        c.minClear, c.minMargin, c.minFeet, ...
        'VariableNames', {'seed', 'fitness', 'speed_mps', 'maxJointVel', 'maxTorque', ...
        'minClear', 'minMargin', 'minFeet'})]; %#ok<AGROW>
end
writetable(results, fullfile(outRoot, 'ga_summary.csv'));
disp(results);
end

function export_gait(r, outDir, g, fcfg, genes)
F = 200;
f2 = fcfg;
f2.nFrames = F;
[~, comp, det] = robot_gait_fitness(r.bestChrom, f2, g);
q = reshape(det.q, F, 8, 3);
table24 = reshape(permute(q, [1 3 2]), F, 24);         % [leg1 h t k, leg2 h t k, ...]
writematrix(round(table24, 7), fullfile(outDir, 'joint_table.csv'));
writematrix(double(reshape(det.stance, F, 8)), fullfile(outDir, 'stance.csv'));

% Standing pose: every foot planted at the centre of its stride.
chrom = r.bestChrom;
T = reshape(det.T, F, 8, 3);

% Start-up: stride grows over 3 steps, then leads into the loop (see
% robot_gait_startup.m). Checked against the same joint limits and speeds.
[Ts, sts] = robot_gait_startup(chrom, F, 3, g);
ns = size(Ts, 1);
legs = repmat(1:8, ns, 1);
qsu = robot_ik(reshape(Ts, ns * 8, 3), legs(:), g);
qsu = reshape(qsu, ns, 8, 3);
writematrix(round(reshape(permute(qsu, [1 3 2]), ns, 24), 7), fullfile(outDir, 'startup.csv'));
writematrix(double(sts), fullfile(outDir, 'startup_stance.csv'));
loopStart = reshape(permute(q(1, :, :), [1 3 2]), 1, 24);
nextAfterStartup = reshape(permute(qsu(end, :, :), [1 3 2]), 1, 24);
suVel = max(abs(diff(reshape(qsu, ns, 24))), [], 'all') * F / chrom(4);
fprintf('  start-up: %d frames, max joint speed %.2f rad/s, last frame -> loop frame 1 gap %.3f rad\n', ...
    ns, suVel, max(abs(loopStart - nextAfterStartup)));
home = [chrom(7) * g.hipX' + chrom(10), chrom(6) * g.side', -chrom(5) * ones(8, 1)];
qs = robot_ik(home, (1:8)', g);
writematrix(round(reshape(qs', 1, 24), 7), fullfile(outDir, 'stand.csv'));

keys = [genes.names, fieldnames(comp)'];
vals = [num2cell(chrom), struct2cell(comp)'];
fid = fopen(fullfile(outDir, 'meta.csv'), 'w');
fprintf(fid, 'key,value\n');
fprintf(fid, 'frames,%d\n', F);
for k = 1:numel(keys)
    fprintf(fid, '%s,%.6g\n', keys{k}, vals{k});
end
fclose(fid);

% Check plots -----------------------------------------------------------------
fig = new_figure(1200, 900);
tl = tiledlayout(fig, 3, 2, 'TileSpacing', 'compact');
t = (0:F - 1) / F * chrom(4);
names = {'hip', 'thigh', 'knee'};
for j = 1:3
    ax = nexttile(tl);
    plot(ax, t, q(:, :, j), 'LineWidth', 1.1);
    hold(ax, 'on');
    yline(ax, g.qMin(j), 'r--');
    yline(ax, g.qMax(j), 'r--');
    title(ax, [names{j} ' angle (rad) with limits']);
    xlabel(ax, 's');
end
legend(ax, g.labels, 'Location', 'eastoutside');
ax = nexttile(tl);
plot(ax, t, max(abs(reshape(det.jointVel, F, 24)), [], 2), 'LineWidth', 1.3);
hold(ax, 'on');
yline(ax, g.maxVel, 'r-', 'motor limit');
yline(ax, fcfg.maxVel, 'r--', 'GA limit');
title(ax, 'fastest joint speed (rad/s)');
xlabel(ax, 's');
ax = nexttile(tl);
imagesc(ax, t, 1:8, ~reshape(det.stance, F, 8)');
colormap(ax, [0 0 0; 1 1 1]);
yticks(ax, 1:8);
yticklabels(ax, g.labels);
title(ax, 'gait diagram (black = planted)');
xlabel(ax, 's');
ax = nexttile(tl);
hold(ax, 'on');
for i = 1:8
    plot(ax, T(:, i, 1), T(:, i, 3), 'LineWidth', 1.2);
end
axis(ax, 'equal');
grid(ax, 'on');
title(ax, 'foot paths, side view (x forward, z up)');
title(tl, sprintf('speed %.3f m/s, joint speed max %.2f rad/s, torque max %.1f Nm, clearance %.3f m', ...
    comp.speed, comp.maxJointVel, comp.maxTorque, comp.minClear));
exportgraphics(fig, fullfile(outDir, 'gait_check.png'), 'Resolution', 100);
close(fig);
end
