function T = run_experiments(seeds)
%RUN_EXPERIMENTS Compare GA components on the gait problem over several seeds.
%   T = run_experiments          % seeds 0..4 (about 10-15 minutes)
%   T = run_experiments(0:2)     % quicker
%
%   Each variant changes ONE thing from the baseline (tournament selection,
%   blend crossover, adaptive mutation, 2 elites, population 100), so any
%   difference can be blamed on that one component. The baseline uses
%   BLEND crossover on purpose: it was the default when these experiments
%   were run, and the default became two-point because of their result.
%
%   Writes spider_cw/outputs/part1/experiments/:
%     summary.csv        one row per variant: mean / std of final fitness and
%                        distance, success rate, generations, run time
%     runs.csv           every single run
%     convergence.png    mean best-fitness curve of each variant, grouped
%                        by component (selection, crossover, mutation and
%                        elitism, population size)
%     final_scores.png   final fitness of every run, per variant
%   A run counts as a SUCCESS when it walks at least 1.0 per cycle, always
%   has >= 4 feet down, and its stability margin never goes negative.

if nargin < 1
    seeds = 0:4;
end
here   = fileparts(mfilename('fullpath'));
outDir = fullfile(here, '..', 'outputs', 'part1', 'experiments');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

variants = {
    'baseline',        'base',      {}
    'roulette',        'selection', {'selection', 'roulette'}
    'rank',            'selection', {'selection', 'rank'}
    'tournament k=5',  'selection', {'tournamentSize', 5}
    'uniform',         'crossover', {'crossover', 'uniform'}
    'two-point',       'crossover', {'crossover', 'twopoint'}
    'no crossover',    'crossover', {'crossoverRate', 0}
    'fixed sigma',     'mutation',  {'adaptiveMutation', false}
    'no elitism',      'mutation',  {'eliteCount', 0}
    'pop 50',          'popsize',   {'popSize', 50}
    'pop 200',         'popsize',   {'popSize', 200}
    };

geom  = spider_geometry();
genes = gait_genes();
fcfg  = fitness_config();
problem.nGenes   = genes.n;
problem.lb       = genes.lb;
problem.ub       = genes.ub;
problem.circular = genes.circular;
problem.evaluate = @(pop) gait_fitness(pop, fcfg, geom);

base = ga_default_config();
base.crossover = 'blend';     % original baseline, see the header
base.verbose = false;
nV = size(variants, 1);
nS = numel(seeds);
curves = cell(nV, 1);
runs = table();

for v = 1:nV
    cfg = base;
    over = variants{v, 3};
    for k = 1:2:numel(over)
        cfg.(over{k}) = over{k + 1};
    end
    curve = nan(nS, cfg.maxGen + 1);
    for i = 1:nS
        cfg.seed = seeds(i);
        r = ga_run(problem, cfg);
        c = r.bestComp;
        ok = c.progress >= 1.0 && c.stanceDeficit == 0 && c.minMargin > 0;
        g = r.history.best';
        curve(i, :) = [g, repmat(g(end), 1, cfg.maxGen + 1 - numel(g))];
        runs = [runs; table(string(variants{v, 1}), seeds(i), r.bestScore, c.progress, ...
            c.minMargin, c.stanceDeficit, ok, r.generations, r.seconds, ...
            'VariableNames', {'variant', 'seed', 'fitness', 'distance', 'minMargin', ...
            'framesShortOfFeet', 'success', 'generations', 'seconds'})]; %#ok<AGROW>
        fprintf('%-15s seed %d: fitness %7.3f distance %5.2f %s  (%d gens, %.1f s)\n', ...
            variants{v, 1}, seeds(i), r.bestScore, c.progress, ...
            ternary(ok, 'walks', 'FAILS'), r.generations, r.seconds);
    end
    curves{v} = curve;
end
writetable(runs, fullfile(outDir, 'runs.csv'));

% Summary per variant.
T = table();
for v = 1:nV
    R = runs(runs.variant == variants{v, 1}, :);
    T = [T; table(string(variants{v, 1}), mean(R.fitness), std(R.fitness), ...
        mean(R.distance), std(R.distance), mean(R.success), mean(R.generations), mean(R.seconds), ...
        'VariableNames', {'variant', 'fitness_mean', 'fitness_std', 'distance_mean', ...
        'distance_std', 'success_rate', 'generations_mean', 'seconds_mean'})]; %#ok<AGROW>
end
writetable(T, fullfile(outDir, 'summary.csv'));
disp(T);

% Convergence curves grouped by component.
groups = {'selection', 'crossover', 'mutation', 'popsize'};
titles = {'Selection', 'Crossover', 'Mutation and elitism', 'Population size'};
fig = new_figure(1200, 800);
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact');
for g = 1:4
    ax = nexttile(tl);
    hold(ax, 'on');
    members = [1; find(strcmp(variants(:, 2), groups{g}))];
    for v = members'
        plot(ax, 0:base.maxGen, mean(curves{v}, 1), 'LineWidth', 1.6, 'DisplayName', variants{v, 1});
    end
    grid(ax, 'on');
    xlabel(ax, 'Generation');
    ylabel(ax, sprintf('Best fitness (mean of %d seeds)', nS));
    title(ax, titles{g});
    legend(ax, 'Location', 'southeast');
end
title(tl, 'Effect of each GA component on the gait problem');
exportgraphics(fig, fullfile(outDir, 'convergence.png'), 'Resolution', 110);
close(fig);

% Final fitness of every run.
fig = new_figure(1100, 500);
ax = axes(fig);
hold(ax, 'on');
for v = 1:nV
    R = runs(runs.variant == variants{v, 1}, :);
    scatter(ax, v * ones(height(R), 1), R.fitness, 36, 'filled', 'MarkerFaceAlpha', 0.6);
    plot(ax, v + [-0.3 0.3], mean(R.fitness) * [1 1], 'k-', 'LineWidth', 2);
end
xticks(ax, 1:nV);
xticklabels(ax, variants(:, 1));
xtickangle(ax, 30);
ylabel(ax, 'Final best fitness');
grid(ax, 'on');
title(ax, sprintf('Final fitness per run (dots) and mean (bar), %d seeds each', nS));
exportgraphics(fig, fullfile(outDir, 'final_scores.png'), 'Resolution', 110);
close(fig);
fprintf('Saved to %s\n', outDir);
end

function out = ternary(cond, a, b)
if cond
    out = a;
else
    out = b;
end
end
