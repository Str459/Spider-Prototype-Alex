function plot_ga_history(result, outDir, label)
%PLOT_GA_HISTORY Plots that show whether the GA is working.
%   fitness.png     best / mean / median score per generation, plus
%                   population diversity and mutation step size sigma.
%                   Healthy run: best rises in steps and never falls
%                   (elitism); mean follows it; diversity shrinks gradually
%                   rather than collapsing to 0 in the first few generations.
%   components.png  every fitness term of the best chromosome per
%                   generation, so you can see WHAT the GA improved
%                   (e.g. first feet onto the ground, then stability,
%                   then stride).

if nargin < 3, label = ''; end
h = result.history;

fig = new_figure(900, 650);
subplot(2, 1, 1);
plot(h.gen, h.best, 'b-', 'LineWidth', 2); hold on;
plot(h.gen, h.mean, 'r-', 'LineWidth', 1);
plot(h.gen, h.median, 'g--', 'LineWidth', 1);
lo = min(h.median);
hi = max(h.best);
ylim([lo - 0.05 * (hi - lo + eps), hi + 0.05 * (hi - lo + eps)]);
grid on;
xlabel('Generation'); ylabel('Fitness');
legend({'best', 'mean', 'median'}, 'Location', 'southeast');
title(sprintf('Fitness progression %s (stopped: %s)', label, result.stopReason), 'Interpreter', 'none');

subplot(2, 1, 2);
yyaxis left;
plot(h.gen, h.diversity, 'LineWidth', 1.5);
ylabel('Diversity (mean gene std / range)');
yyaxis right;
plot(h.gen, h.sigma, 'LineWidth', 1.5);
ylabel('Mutation sigma');
grid on;
xlabel('Generation');
title('Population diversity and adaptive mutation step');
exportgraphics(fig, fullfile(outDir, 'fitness.png'), 'Resolution', 110);
close(fig);

names = setdiff(fieldnames(h.bestComp), {'score'}, 'stable');
fig = new_figure(1200, 750);
tl = tiledlayout(fig, 3, 4, 'TileSpacing', 'compact');
for k = 1:numel(names)
    nexttile(tl);
    plot(h.gen, h.bestComp.(names{k}), 'LineWidth', 1.5);
    grid on;
    title(names{k});
    xlabel('gen');
end
title(tl, ['Fitness terms of the best chromosome ' label], 'Interpreter', 'none');
exportgraphics(fig, fullfile(outDir, 'components.png'), 'Resolution', 110);
close(fig);
end
