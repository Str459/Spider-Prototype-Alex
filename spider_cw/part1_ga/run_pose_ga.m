function result = run_pose_ga(varargin)
%RUN_POSE_GA Stage A: evolve one stable standing pose (a single 1x24 vector).
%   run_pose_ga                       % seed 0
%   run_pose_ga('seed', 2, 'popSize', 60)
%   This is the most direct reading of the spec: the chromosome is exactly
%   the 24 joint angles that plot_spider_pose accepts. It uses the same GA
%   engine and operators as the gait (Stage B), and is a quick way to see
%   them working on a problem whose answer is easy to judge by eye.
%
%   Writes spider_cw/outputs/part1/pose_seed<seed>/: best_pose.csv (1 x 24),
%   fitness.png, components.png, pose.png, lecturer_view.png, summary.txt

here = fileparts(mfilename('fullpath'));
cfg  = ga_default_config();
cfg.maxGen = 200;
fcfg = fitness_config();
name = '';
for k = 1:2:numel(varargin)
    key = varargin{k};
    val = varargin{k + 1};
    if isfield(cfg, key)
        cfg.(key) = val;
    elseif isfield(fcfg, key)
        fcfg.(key) = val;
    elseif strcmp(key, 'name')
        name = val;
    else
        error('run_pose_ga:unknownOption', 'Unknown option "%s"', key);
    end
end
if isempty(name)
    name = sprintf('pose_seed%d', cfg.seed);
end
outDir = fullfile(here, '..', 'outputs', 'part1', name);
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

geom = spider_geometry();
% Same joint ranges as the gait genes: coxa +-0.8, femur [-1.2, 0.6], tibia [-2, 0].
problem.nGenes   = 24;
problem.lb       = repmat([-0.8, -1.2, -2.0], 1, 8);
problem.ub       = repmat([ 0.8,  0.6,  0.0], 1, 8);
problem.circular = false(1, 24);
problem.evaluate = @(pop) pose_fitness(pop, fcfg, geom);

fprintf('=== Pose GA "%s" ===\n', name);
result = ga_run(problem, cfg);
pose = result.bestChrom;
writematrix(round(pose, 8), fullfile(outDir, 'best_pose.csv'));

fid = fopen(fullfile(outDir, 'summary.txt'), 'w');
fprintf(fid, 'Pose GA "%s": %d generations, %s, %.1f s\n', name, result.generations, result.stopReason, result.seconds);
f = fieldnames(result.bestComp);
for k = 1:numel(f)
    fprintf(fid, '  %-12s %8.4f\n', f{k}, result.bestComp.(f{k}));
end
fprintf(fid, '\nBest pose (radians, spec order):\n');
for leg = 1:8
    fprintf(fid, '  %s  a %7.3f  b %7.3f  c %7.3f\n', geom.labels{leg}, pose(3 * leg - 2 : 3 * leg));
end
fclose(fid);

plot_ga_history(result, outDir, name);

fig = new_figure(700, 600);
ax = axes(fig);
hold(ax, 'on');
fill3(ax, [-4 4 4 -4], [-4 -4 4 4], fcfg.groundZ * [1 1 1 1], [0.93 0.95 0.9], 'EdgeColor', 'none');
draw_spider(ax, pose, geom);
axis(ax, 'equal'); grid(ax, 'on'); view(ax, 35, 25);
xlim(ax, [-4 4]); ylim(ax, [-4 4]); zlim(ax, [fcfg.groundZ - 0.2, 1.5]);
title(ax, sprintf('Evolved standing pose (margin %.2f, ground error %.4f)', ...
    result.bestComp.margin, result.bestComp.groundErr));
exportgraphics(fig, fullfile(outDir, 'pose.png'), 'Resolution', 110);
close(fig);

addpath(fullfile(here, '..', '..', 'AI cw'));
evalc('plot_spider_pose(pose)');
fig = figure(1);
try
    theme(fig, 'light');
catch
end
title('Evolved pose drawn by the provided plot\_spider\_pose.m');
exportgraphics(fig, fullfile(outDir, 'lecturer_view.png'), 'Resolution', 100);
close(fig);
fprintf('Saved to %s\n', outDir);
end
