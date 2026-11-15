function kids = ga_mutation(kids, sigma, cfg, problem)
%GA_MUTATION Gaussian mutation, then repair genes back into their bounds.
%   Each gene mutates with probability cfg.mutationRate by adding
%   N(0, (sigma * gene range)^2). Scaling by the range means one sigma
%   means the same for a phase gene (range 1) and a tibia gene (range 2 rad).
%   sigma itself is adapted by ga_run (bigger when stuck, smaller when
%   improving).
%
%   Repair: circular genes (phases) wrap round; all others are clipped to
%   their bounds. This also repairs children that blend crossover pushed
%   slightly outside the bounds.

[M, n] = size(kids);
span = problem.ub - problem.lb;
hit = rand(M, n) < cfg.mutationRate;
kids = kids + hit .* randn(M, n) .* (sigma * span);
kids = repair(kids, problem);
end

function x = repair(x, problem)
lb = problem.lb;
ub = problem.ub;
circ = problem.circular;
span = ub - lb;
x(:, circ) = lb(circ) + mod(x(:, circ) - lb(circ), span(circ));
x(:, ~circ) = min(max(x(:, ~circ), lb(~circ)), ub(~circ));
end
