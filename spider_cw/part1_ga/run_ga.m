function best = run_ga(selectionType, crossoverType, seed)
% Part 1: a genetic algorithm that evolves a walking gait for the spider.
%   run_ga                            default settings (tournament, two-point, seed 1)
%   run_ga('roulette', 'uniform', 2)  other selection / crossover / random seed
% Results are saved in ../outputs/part1/<selection>_<crossover>_seed<seed>/

if nargin < 1, selectionType = 'tournament'; end   % 'tournament' or 'roulette'
if nargin < 2, crossoverType = 'twopoint'; end     % 'twopoint' or 'uniform'
if nargin < 3, seed = 1; end

% GA settings
popSize = 60;
maxGen = 200;
stallLimit = 40;       % stop early if the best hasn't improved for this many generations
eliteCount = 2;        % best chromosomes copied unchanged into the next generation
tournamentSize = 3;
crossoverRate = 0.9;
mutationRate = 0.1;    % chance that each gene gets mutated
mutationStep = 0.1;    % size of a mutation (std) as a fraction of the gene's range

% Fitness settings (see gait_fitness.m)
s.nFrames = 300;
s.groundZ = -0.4;           % ground height below the body
s.clearance = 0.15;         % how high a swinging foot must lift
s.maxLift = 0.5;            % and the most it should lift
s.maxGap = deg2rad(150);    % stability limit
s.footGap = 0.6;            % closest two neighbouring feet may get
s.weights = struct('distance', 3, 'slip', 1, 'ground', 20, 'below', 30, ...
                   'lift', 10, 'unstable', 50, 'feet', 20);

% Gene limits: six per leg pair (see decode_gait.m), then phaseLR, phaseFB, duty
pairLow  = [-0.5 0   -1.2 0   -2 -0.8];
pairHigh = [ 0.5 0.5  0.6 0.8  0  0.8];
low  = [repmat(pairLow, 1, 4), 0, 0, 0.5];
high = [repmat(pairHigh, 1, 4), 1, 1, 0.9];
nGenes = numel(low);

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', '..', 'AI cw'));     % lecturer's files
outDir = fullfile(here, '..', 'outputs', 'part1', sprintf('%s_%s_seed%d', selectionType, crossoverType, seed));
if ~exist(outDir, 'dir'), mkdir(outDir); end

%% The GA
rng(seed);
pop = low + rand(popSize, nGenes) .* (high - low);   % random starting population
fit = evaluate(pop, s);
bestHist = [];
meanHist = [];
stall = 0;
tic;
for gen = 1:maxGen
    [fit, order] = sort(fit, 'descend');
    pop = pop(order, :);
    bestHist(gen) = fit(1); %#ok<AGROW>
    meanHist(gen) = mean(fit); %#ok<AGROW>
    if gen > 1 && bestHist(gen) < bestHist(gen - 1) + 1e-4
        stall = stall + 1;
    else
        stall = 0;
    end
    if mod(gen, 10) == 1
        fprintf('gen %3d   best %7.3f   mean %8.3f\n', gen, fit(1), mean(fit));
    end
    if stall >= stallLimit
        fprintf('stopped: no improvement for %d generations\n', stallLimit);
        break
    end

    % make the next generation
    kids = zeros(popSize - eliteCount, nGenes);
    for k = 1:2:size(kids, 1)
        p1 = pop(selectParent(fit, selectionType, tournamentSize), :);
        p2 = pop(selectParent(fit, selectionType, tournamentSize), :);
        if rand < crossoverRate
            [c1, c2] = crossover(p1, p2, crossoverType);
        else
            c1 = p1;
            c2 = p2;
        end
        kids(k, :) = mutate(c1, mutationRate, mutationStep, low, high);
        if k + 1 <= size(kids, 1)
            kids(k + 1, :) = mutate(c2, mutationRate, mutationStep, low, high);
        end
    end
    pop = [pop(1:eliteCount, :); kids];                % elitism: the best few survive
    fit = [fit(1:eliteCount); evaluate(kids, s)];
end
fprintf('%d generations in %.1f s\n', gen, toc);

%% Results
[~, i] = max(fit);
best = pop(i, :);
[score, info] = gait_fitness(best, s);
[gait, down] = decode_gait(best, s.nFrames);
[x, y, z] = foot_positions(gait);
fprintf('best fitness %.3f\n', score);
fprintf('distance per cycle %.2f, slip %.2f, unstable frames %.1f%%, feet down %d to %d\n', ...
    info.distance, info.slip, 100 * info.unstable, min(sum(down, 2)), max(sum(down, 2)));

writematrix(gait, fullfile(outDir, 'best_gait.csv'));          % 300 x 24, one pose per row
writematrix(best, fullfile(outDir, 'best_chromosome.csv'));
writematrix([(1:numel(bestHist))', bestHist', meanHist'], fullfile(outDir, 'fitness_history.csv'));

% check foot_positions.m against the lecturer's forward_leg_kinematics2.m
base = deg2rad([45 75 105 135 -135 -105 -75 -45]);
err = 0;
for f = 1:10:s.nFrames
    for leg = 1:8
        [~, ~, ~, foot] = forward_leg_kinematics2([1.5 * cos(base(leg)), sin(base(leg)), 0], ...
            base(leg), gait(f, 3 * leg - 2:3 * leg), [1.2 0.7 1.0]);
        err = max(err, norm(foot - [x(f, leg), y(f, leg), z(f, leg)]));
    end
end
fprintf('biggest difference from the lecturer''s kinematics: %.1e\n', err);

% fitness over generations
figure(2); clf; lightTheme(2);
plot(bestHist, 'LineWidth', 2); hold on;
plot(meanHist);
ylim([min(bestHist) - 5, max(bestHist) + 1]);
xlabel('Generation'); ylabel('Fitness'); grid on;
legend('Best', 'Population mean', 'Location', 'southeast');
title('Fitness over generations');
saveas(figure(2), fullfile(outDir, 'fitness.png'));

% foot heights and which feet are down
names = {'L1', 'L2', 'L3', 'L4', 'R4', 'R3', 'R2', 'R1'};
figure(3); clf; lightTheme(3);
subplot(2, 1, 1);
plot(z - s.groundZ); grid on;
xlabel('Frame'); ylabel('Height above ground');
legend(names, 'Location', 'eastoutside');
title('Foot heights over one cycle');
subplot(2, 1, 2);
imagesc(~down'); colormap(gray);
yticks(1:8); yticklabels(names); xlabel('Frame');
title('Feet on the ground (black)');
saveas(figure(3), fullfile(outDir, 'feet.png'));

% first pose and an animation, drawn with the lecturer's plot_spider_pose
% (evalc hides the angles it prints for every frame)
gifFile = fullfile(outDir, 'gait.gif');
lightTheme(1);
for f = 1:5:s.nFrames
    evalc('plot_spider_pose(gait(f, :))');
    title(sprintf('Frame %d of %d', f, s.nFrames));
    drawnow;
    if f == 1
        saveas(figure(1), fullfile(outDir, 'pose.png'));
    end
    [im, map] = rgb2ind(frame2im(getframe(figure(1))), 256);
    if f == 1
        imwrite(im, map, gifFile, 'LoopCount', Inf, 'DelayTime', 0.05);
    else
        imwrite(im, map, gifFile, 'WriteMode', 'append', 'DelayTime', 0.05);
    end
end
fprintf('saved results in %s\n', outDir);
end


function fit = evaluate(pop, s)
fit = zeros(size(pop, 1), 1);
for i = 1:size(pop, 1)
    fit(i) = gait_fitness(pop(i, :), s);
end
end


function i = selectParent(fit, type, k)
if strcmp(type, 'tournament')
    % pick k chromosomes at random, the fittest one wins
    entrants = randi(numel(fit), k, 1);
    [~, j] = max(fit(entrants));
    i = entrants(j);
else
    % roulette wheel: chance proportional to fitness
    % (shifted so every weight is positive, since fitness can be negative)
    w = fit - min(fit) + 1e-6;
    i = find(rand * sum(w) <= cumsum(w), 1);
end
end


function [c1, c2] = crossover(p1, p2, type)
c1 = p1;
c2 = p2;
if strcmp(type, 'twopoint')
    % swap the block of genes between two random cut points
    cuts = sort(randperm(numel(p1), 2));
    swap = cuts(1):cuts(2);
else
    % uniform: each gene is swapped with a 50% chance
    swap = rand(size(p1)) < 0.5;
end
c1(swap) = p2(swap);
c2(swap) = p1(swap);
end


function c = mutate(c, rate, step, low, high)
% Gaussian mutation: each gene has a small chance of a random nudge
hit = rand(size(c)) < rate;
c = c + hit .* randn(size(c)) .* step .* (high - low);
c = min(max(c, low), high);     % keep genes inside their limits
end


function lightTheme(n)
% newer MATLAB versions draw dark figures by default; older ones have no theme()
try
    theme(figure(n), 'light');
catch
end
end
