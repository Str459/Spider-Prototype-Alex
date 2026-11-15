function result = run_gait_ga(varargin)
%RUN_GAIT_GA Evolve a walking gait and save everything (Part 1, Stage B).
%   Run from the spider_cw/part1_ga folder in MATLAB, e.g.
%       run_gait_ga                                   % seed 0, defaults
%       run_gait_ga('seed', 3)                        % another seed
%       run_gait_ga('selection', 'roulette', 'name', 'roulette_s0')
%       run_gait_ga('targetSpeed', 2.0, 'name', 'speed_2.0')
%       run_gait_ga('w_slip', 3)                      % change a fitness weight
%   Any field of ga_default_config or fitness_config can be given by name;
%   fitness weights use the prefix "w_". Other options:
%       'name'  output folder name (default 'seed<seed>')
%       'plots' false to skip figures, 'gif' false to skip the animation
%
%   Writes to spider_cw/outputs/part1/<name>/:
%       best_gait.csv         300 x 24 joint angles, radians, one pose per
%                             row in the spec's order [L1a L1b L1c ... R1c].
%                             Each row can be passed to plot_spider_pose.
%       best_gait_stance.csv  300 x 8, 1 = foot planted (same leg order)
%       best_chrom.csv        the 27 genes of the best chromosome
%       history.csv           per-generation statistics
%       summary.txt           settings, result, every fitness term
%       result.mat            everything above as MATLAB variables
%       fitness.png, components.png, joint_angles.png, foot_heights.png,
%       stability.png, pose_snapshots.png, gait.gif, lecturer_view.png

here = fileparts(mfilename('fullpath'));
cfg  = ga_default_config();
fcfg = fitness_config();
opts = struct('name', '', 'outRoot', fullfile(here, '..', 'outputs', 'part1'), ...
              'plots', true, 'gif', true);

for k = 1:2:numel(varargin)
    key = varargin{k};
    val = varargin{k + 1};
    if isfield(cfg, key)
        cfg.(key) = val;
    elseif isfield(fcfg, key)
        fcfg.(key) = val;
    elseif startsWith(key, 'w_') && isfield(fcfg.w, key(3:end))
        fcfg.w.(key(3:end)) = val;
    elseif isfield(opts, key)
        opts.(key) = val;
    else
        error('run_gait_ga:unknownOption', 'Unknown option "%s"', key);
    end
end
if isempty(opts.name)
    opts.name = sprintf('seed%d', cfg.seed);
end
outDir = fullfile(opts.outRoot, opts.name);
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

geom  = spider_geometry();
genes = gait_genes();
problem.nGenes   = genes.n;
problem.lb       = genes.lb;
problem.ub       = genes.ub;
problem.circular = genes.circular;
problem.evaluate = @(pop) gait_fitness(pop, fcfg, geom);

fprintf('=== Gait GA "%s": pop %d, %s selection, %s crossover, seed %d ===\n', ...
    opts.name, cfg.popSize, cfg.selection, cfg.crossover, cfg.seed);
result = ga_run(problem, cfg);

% Decode the winner at full resolution and keep the per-frame details.
[~, comp, detail] = gait_fitness(result.bestChrom, fcfg, geom);
gait   = reshape(detail.angles, fcfg.nFrames, 24);
stance = reshape(detail.stance, fcfg.nFrames, 8);

writematrix(round(gait, 8), fullfile(outDir, 'best_gait.csv'));
writematrix(double(stance), fullfile(outDir, 'best_gait_stance.csv'));
writetable(table(genes.names', result.bestChrom', 'VariableNames', {'gene', 'value'}), ...
    fullfile(outDir, 'best_chrom.csv'));
writetable(history_table(result.history), fullfile(outDir, 'history.csv'));
write_summary(fullfile(outDir, 'summary.txt'), opts.name, result, comp, cfg, fcfg, genes, detail);
save(fullfile(outDir, 'result.mat'), 'result', 'comp', 'detail', 'cfg', 'fcfg', 'genes');

if opts.plots
    plot_ga_history(result, outDir, opts.name);
    plot_gait_analysis(detail, geom, fcfg, outDir);
    animate_gait(detail, geom, fcfg, outDir, 'gif', opts.gif);
    lecturer_view(gait(1, :), here, outDir);
end
fprintf('Distance per cycle %.3f, min stability margin %.3f, frames with < %d feet: %.1f%%\n', ...
    comp.progress, comp.minMargin, fcfg.minStance, 100 * comp.stanceDeficit);
fprintf('Saved to %s\n', outDir);
end

function T = history_table(h)
T = table(h.gen, h.best, h.mean, h.median, h.worst, h.std, h.diversity, h.sigma, ...
    'VariableNames', {'generation', 'best', 'mean', 'median', 'worst', 'std', 'diversity', 'sigma'});
names = fieldnames(h.bestComp);
for k = 1:numel(names)
    T.(['best_' names{k}]) = h.bestComp.(names{k});
end
end

function lecturer_view(pose, here, outDir)
% Draw the first frame with the lecturer's own plot_spider_pose, to show
% the exported angles work with the provided model unchanged.
addpath(fullfile(here, '..', '..', 'AI cw'));
evalc('plot_spider_pose(pose)');
fig = figure(1);
try
    theme(fig, 'light');
catch
end
title('Frame 1 of best\_gait.csv drawn by the provided plot\_spider\_pose.m');
exportgraphics(fig, fullfile(outDir, 'lecturer_view.png'), 'Resolution', 100);
close(fig);
end

function write_summary(file, name, result, comp, cfg, fcfg, genes, detail)
fid = fopen(file, 'w');
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, 'Gait GA run "%s"\n', name);
fprintf(fid, 'Stopped after %d generations: %s (%.1f s)\n\n', result.generations, result.stopReason, result.seconds);
fprintf(fid, 'RESULT (best chromosome, scored on %d frames)\n', fcfg.nFrames);
names = fieldnames(comp);
for k = 1:numel(names)
    fprintf(fid, '  %-18s %10.4f\n', names{k}, comp.(names{k}));
end
fprintf(fid, '  %-18s %10d\n', 'fewest feet down', min(detail.nStance(:)));
fprintf(fid, '\nGENES\n');
for k = 1:genes.n
    fprintf(fid, '  %-22s %8.4f   [%g, %g]\n', genes.names{k}, result.bestChrom(k), genes.lb(k), genes.ub(k));
end
fprintf(fid, '\nGA SETTINGS\n');
f = fieldnames(cfg);
for k = 1:numel(f)
    fprintf(fid, '  %-18s %s\n', f{k}, num2str(cfg.(f{k})));
end
fprintf(fid, '\nFITNESS SETTINGS\n');
f = setdiff(fieldnames(fcfg), {'w'}, 'stable');
for k = 1:numel(f)
    fprintf(fid, '  %-18s %s\n', f{k}, num2str(fcfg.(f{k})));
end
f = fieldnames(fcfg.w);
for k = 1:numel(f)
    fprintf(fid, '  w.%-16s %s\n', f{k}, num2str(fcfg.w.(f{k})));
end
end
