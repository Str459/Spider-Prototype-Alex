function [c1, c2] = ga_crossover(p1, p2, cfg, problem)
%GA_CROSSOVER Recombine pairs of parents (row i of p1 with row i of p2).
%   Each pair is recombined with probability crossoverRate, otherwise the
%   children are copies of the parents. Children may fall slightly outside
%   the gene bounds; ga_mutation repairs them.
%
%   'blend' (BLX-alpha, default): every gene of a child is drawn uniformly
%       from the parents' interval widened by alpha on each side. Suits
%       real-valued genes: children can land BETWEEN parents (e.g. a stride
%       half-way between two good ones), not just copy one or the other.
%       alpha > 0 keeps the population from shrinking too fast.
%   'uniform': each gene copied from either parent with a 50/50 coin.
%   'onepoint' / 'twopoint': swap one or two contiguous blocks of genes.
%       Our genes are grouped per leg pair, so block crossover tends to
%       keep a whole leg's settings together.
%
%   Phase genes are circular (0.95 and 0.05 are only 0.1 apart). Before
%   blending, the second parent's phase is moved by a whole cycle if that
%   brings it closer, so the child lands between them the short way round.

[M, n] = size(p1);
c1 = p1;
c2 = p2;

span = problem.ub - problem.lb;
circ = problem.circular;
q2 = p2;
d = q2(:, circ) - p1(:, circ);
d = d - span(circ) .* round(d ./ span(circ));
q2(:, circ) = p1(:, circ) + d;

switch lower(cfg.crossover)
    case 'blend'
        a  = cfg.blendAlpha;
        lo = min(p1, q2);
        hi = max(p1, q2);
        I  = hi - lo;
        c1 = lo - a * I + rand(M, n) .* (I * (1 + 2 * a));
        c2 = lo - a * I + rand(M, n) .* (I * (1 + 2 * a));

    case 'uniform'
        m = rand(M, n) < 0.5;
        c1(m) = p2(m);
        c2(m) = p1(m);

    case 'onepoint'
        cut = randi(n - 1, M, 1);
        m = (1:n) > cut;
        c1(m) = p2(m);
        c2(m) = p1(m);

    case 'twopoint'
        a = randi(n, M, 1);
        b = randi(n, M, 1);
        lo = min(a, b);
        hi = max(a, b);
        m = (1:n) >= lo & (1:n) <= hi;
        c1(m) = p2(m);
        c2(m) = p1(m);

    otherwise
        error('ga_crossover:unknown', 'Unknown crossover "%s"', cfg.crossover);
end

keep = rand(M, 1) >= cfg.crossoverRate;   % these pairs are not recombined
c1(keep, :) = p1(keep, :);
c2(keep, :) = p2(keep, :);
end
