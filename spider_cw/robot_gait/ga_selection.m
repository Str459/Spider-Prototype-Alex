function idx = ga_selection(score, nPick, cfg)
%GA_SELECTION Choose parents. Returns nPick indices into the population.
%   'tournament' (default): pick tournamentSize chromosomes at random and
%       keep the best. Only the ORDER of scores matters, so it works with
%       negative scores and a single outstanding chromosome cannot take
%       over the population in one generation. Pressure is tuned with
%       tournamentSize (2 = gentle, 5+ = greedy).
%   'roulette': chance of being picked proportional to fitness. Our scores
%       can be negative, so they are shifted so the worst is ~0 ("windowing").
%       Sensitive to scale: early on one lucky chromosome can dominate;
%       late on, when scores are close, pressure almost disappears.
%   'rank': linear ranking. Chance proportional to rank (best = N, worst = 1),
%       which keeps roulette's idea but removes the scale problem.

score = score(:);
N = numel(score);
switch lower(cfg.selection)
    case 'tournament'
        k = cfg.tournamentSize;
        cand = randi(N, nPick, k);
        [~, w] = max(score(cand), [], 2);
        idx = cand(sub2ind([nPick k], (1:nPick)', w));

    case 'roulette'
        f = score - min(score);
        f = f + 1e-9 + 1e-3 * mean(f);          % worst still has a tiny chance
        idx = spin(f, nPick);

    case 'rank'
        [~, order] = sort(score, 'ascend');
        r = zeros(N, 1);
        r(order) = 1:N;                          % worst = 1, best = N
        idx = spin(r, nPick);

    otherwise
        error('ga_selection:unknown', 'Unknown selection "%s"', cfg.selection);
end
end

function idx = spin(weights, nPick)
% Spin a roulette wheel whose slices are proportional to weights.
edges = cumsum(weights) / sum(weights);
idx = sum(rand(nPick, 1) > edges', 2) + 1;
idx = min(idx, numel(weights));
end
