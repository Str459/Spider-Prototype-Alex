function cfg = ga_default_config()
%GA_DEFAULT_CONFIG Every tunable parameter of the genetic algorithm.
%   Override any field by name, for example
%       run_gait_ga('popSize', 200, 'selection', 'roulette', 'seed', 3)
%   See docs/part1_operators.txt for why these defaults were chosen.

% Population and stopping
cfg.popSize          = 100;     % chromosomes per generation
cfg.maxGen           = 250;     % hard limit on generations
cfg.stallGenerations = 80;      % stop after this many generations without improvement
cfg.targetFitness    = Inf;     % stop early once the best score reaches this
cfg.tolImprove       = 1e-6;    % smaller gains do not count as improvement
cfg.eliteCount       = 2;       % best chromosomes copied unchanged (elitism)
cfg.seed             = 0;       % random seed, so every run can be repeated

% Selection: 'tournament' | 'roulette' | 'rank'
cfg.selection        = 'tournament';
cfg.tournamentSize   = 3;

% Crossover: 'twopoint' | 'blend' (BLX-alpha) | 'uniform' | 'onepoint'
% Two-point won the operator experiments (outputs/part1/experiments):
% genes are grouped per leg pair, so swapping blocks passes on whole legs.
cfg.crossover        = 'twopoint';
cfg.crossoverRate    = 0.9;     % chance a pair of parents is recombined
cfg.blendAlpha       = 0.3;     % BLX-alpha: how far past the parents a child may land

% Mutation: Gaussian noise on each gene with probability mutationRate.
% Step size sigma is a fraction of each gene's range.
cfg.mutationRate     = 0.15;
cfg.sigma0           = 0.10;    % starting step size
cfg.adaptiveMutation = true;    % false = sigma stays at sigma0
cfg.sigmaMin         = 0.01;
cfg.sigmaMax         = 0.15;    % 0.30 scattered the whole population once stalled
cfg.sigmaDecay       = 0.98;    % shrink after an improvement (fine-tune)
cfg.sigmaGrow        = 1.3;     % grow after sigmaPatience stalled generations (explore)
cfg.sigmaPatience    = 15;

% Output
cfg.verbose          = true;
cfg.printEvery       = 10;
end
