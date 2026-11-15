function result = ga_run(problem, cfg)
%GA_RUN A complete, problem-independent real-valued genetic algorithm.
%   result = GA_RUN(problem, cfg)
%     problem.nGenes    number of genes
%     problem.lb/ub     1 x nGenes bounds
%     problem.circular  1 x nGenes logical, genes that wrap round (phases)
%     problem.evaluate  function handle: [score, comp] = evaluate(pop)
%                       pop is K x nGenes, score K x 1 (higher = better),
%                       comp a struct of K x 1 vectors (fitness breakdown)
%     cfg               settings from ga_default_config.m
%
%   The same engine runs the single-pose problem (Stage A, 24 genes) and
%   the gait problem (Stage B, 27 genes), and a test function in
%   test_part1.m. It knows nothing about spiders.
%
%   Loop:
%     1. Initialise: popSize random chromosomes, uniform within the bounds.
%     2. Evaluate every chromosome.
%     3. Repeat until a stopping rule fires:
%          sort by score; record statistics;
%          keep the eliteCount best unchanged (elitism, so the best
%            solution found can never be lost);
%          select parents (ga_selection), recombine (ga_crossover),
%            mutate and repair (ga_mutation) to fill the rest;
%          evaluate the children;
%          adapt the mutation step size sigma.
%     Stopping rules: maxGen reached, no improvement for stallGenerations,
%     or best score >= targetFitness.
%
%   result fields: bestChrom, bestScore, bestComp, history (per-generation
%   statistics), finalPop, finalScore, stopReason, generations, seconds, cfg.

rng(cfg.seed, 'twister');
n    = problem.nGenes;
lb   = problem.lb;
ub   = problem.ub;
span = ub - lb;
N    = cfg.popSize;
E    = min(cfg.eliteCount, N - 1);

% 1-2. Random initial population, evaluated.
pop = lb + rand(N, n) .* span;
[score, comp] = problem.evaluate(pop);
compNames = fieldnames(comp);

H = cfg.maxGen + 1;
hist.gen       = nan(H, 1);
hist.best      = nan(H, 1);
hist.mean      = nan(H, 1);
hist.median    = nan(H, 1);
hist.worst     = nan(H, 1);
hist.std       = nan(H, 1);
hist.diversity = nan(H, 1);
hist.sigma     = nan(H, 1);
for f = 1:numel(compNames)
    hist.bestComp.(compNames{f}) = nan(H, 1);
end

sigma = cfg.sigma0;
bestScore = -Inf;
stall = 0;
stopReason = 'maxGen';
timer = tic;

for gen = 0:cfg.maxGen
    % Sort so the best chromosome is row 1.
    [score, order] = sort(score, 'descend');
    pop = pop(order, :);
    for f = 1:numel(compNames)
        comp.(compNames{f}) = comp.(compNames{f})(order);
    end

    % Remember the best chromosome ever seen. With elitism it is always
    % row 1 anyway; with eliteCount = 0 it could otherwise be lost.
    improved = score(1) > bestScore + cfg.tolImprove;
    if score(1) > bestScore
        bestScore = score(1);
        bestChrom = pop(1, :);
        bestComp  = structfun(@(v) v(1), comp, 'UniformOutput', false);
    end
    if improved
        stall = 0;
    else
        stall = stall + 1;
    end

    % Record statistics for the plots.
    h = gen + 1;
    hist.gen(h)       = gen;
    hist.best(h)      = score(1);      % best of THIS generation (can drop without elitism)
    hist.mean(h)      = mean(score);
    hist.median(h)    = median(score);
    hist.worst(h)     = score(end);
    hist.std(h)       = std(score);
    hist.diversity(h) = mean(std(pop, 0, 1) ./ span);   % 0 = everyone identical
    hist.sigma(h)     = sigma;
    for f = 1:numel(compNames)
        hist.bestComp.(compNames{f})(h) = comp.(compNames{f})(1);
    end

    if cfg.verbose && (mod(gen, cfg.printEvery) == 0 || gen == cfg.maxGen)
        fprintf('gen %4d | best %8.4f | mean %8.4f | diversity %.3f | sigma %.3f\n', ...
            gen, score(1), hist.mean(h), hist.diversity(h), sigma);
    end

    % Stopping rules.
    if gen == cfg.maxGen
        stopReason = 'maxGen';
        break
    elseif stall >= cfg.stallGenerations
        stopReason = sprintf('no improvement for %d generations', cfg.stallGenerations);
        break
    elseif score(1) >= cfg.targetFitness
        stopReason = 'targetFitness reached';
        break
    end

    % Adapt the mutation step size.
    if cfg.adaptiveMutation
        if improved
            sigma = max(cfg.sigmaMin, sigma * cfg.sigmaDecay);
        elseif mod(stall, cfg.sigmaPatience) == 0
            sigma = min(cfg.sigmaMax, sigma * cfg.sigmaGrow);
        end
    end

    % Breed the next generation.
    nKids  = N - E;
    nPairs = ceil(nKids / 2);
    parents = ga_selection(score, 2 * nPairs, cfg);
    [c1, c2] = ga_crossover(pop(parents(1:2:end), :), pop(parents(2:2:end), :), cfg, problem);
    kids = [c1; c2];
    kids = kids(1:nKids, :);
    kids = ga_mutation(kids, sigma, cfg, problem);
    [kidScore, kidComp] = problem.evaluate(kids);

    pop   = [pop(1:E, :); kids];
    score = [score(1:E); kidScore];
    for f = 1:numel(compNames)
        comp.(compNames{f}) = [comp.(compNames{f})(1:E); kidComp.(compNames{f})];
    end
end

% Trim unused history rows.
keep = ~isnan(hist.gen);
for f = {'gen', 'best', 'mean', 'median', 'worst', 'std', 'diversity', 'sigma'}
    hist.(f{1}) = hist.(f{1})(keep);
end
for f = 1:numel(compNames)
    hist.bestComp.(compNames{f}) = hist.bestComp.(compNames{f})(keep);
end

result.bestChrom   = bestChrom;
result.bestScore   = bestScore;
result.bestComp    = bestComp;
result.history     = hist;
result.finalPop    = pop;
result.finalScore  = score;
result.stopReason  = stopReason;
result.generations = hist.gen(end);
result.seconds     = toc(timer);
result.cfg         = cfg;

if cfg.verbose
    fprintf('Stopped after %d generations (%s), %.1f s. Best score %.4f\n', ...
        result.generations, stopReason, result.seconds, result.bestScore);
end
end
