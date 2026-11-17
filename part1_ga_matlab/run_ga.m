function best = run_ga(selectionType, crossoverType, seed, targetSpeed)
% Part 1: a genetic algorithm that evolves a walking gait for the Webots spider robot.
%   run_ga                                    default (tournament, two-point, seed 1): fastest gait
%   run_ga('roulette', 'uniform', 2)          other selection / crossover / random seed
%   run_ga('tournament', 'twopoint', 1, 0.2)  a gait that walks at 0.2 m/s (training data for Part 2)
% Results are saved in ../outputs/part1/<selection>_<crossover>_seed<seed>/ or ../outputs/part1/speed_<v>/
% To walk a result in Webots open ../webots/worlds/spider.wbt (see README.md).

if nargin < 1, selectionType = 'tournament'; end   % 'tournament' or 'roulette'
if nargin < 2, crossoverType = 'twopoint'; end     % 'twopoint' or 'uniform'
if nargin < 3, seed = 1; end
if nargin < 4, targetSpeed = NaN; end              % NaN = as fast as possible

% GA settings
popSize = 60;
maxGen = 200;
stallLimit = 40;       % stop early if the best hasn't improved for this many generations
eliteCount = 2;        % best chromosomes copied unchanged into the next generation
tournamentSize = 3;
crossoverRate = 0.9;
mutationRate = 0.1;    % chance that each gene gets mutated
mutationStep = 0.1;    % size of a mutation (std) as a fraction of the gene's range

% Fitness settings (see gait_fitness.m). The robot's limits come from protos/SpiderOctopod.proto.
s.nFrames = 300;
s.jointMin = [-1.5708 -1.5708 -2];   % hip, thigh, knee motor limits (rad)
s.jointMax = [ 1.5708  1.5708  1];
s.margin = 0.05;                     % stay this far inside the limits
s.maxJointSpeed = 1.6;               % rad/s (the motors can do 2)
s.legGap = 0.15;                     % closest two neighbouring legs may get (m)
s.maxGap = deg2rad(150);             % stability limit
s.weights = struct('speed', 10, 'reach', 50, 'limits', 20, 'fast', 5, 'legs', 50, 'unstable', 20, ...
                   'target', 50, 'change', 2);

% Gene limits (see decode_gait.m):
%        stride stepHeight duty period bodyHeight reach spread xShift
low  = [ 0.05   0.06       0.52 1.5    0.42       0.65  1.0    -0.15];
high = [ 0.45   0.20       0.75 6.0    0.60       1.00  2.2     0.15];
nGenes = numel(low);

here = fileparts(mfilename('fullpath'));
outName = sprintf('%s_%s_seed%d', selectionType, crossoverType, seed);
s.targetSpeed = targetSpeed;
if ~isnan(targetSpeed)
    % Gait for one speed: start from the fastest gait (run_ga with no arguments), keep its
    % period and duty, and change it as little as possible (gait_fitness.m). Then every
    % gait in the speed family has the same timing and neighbouring speeds are similar
    % gaits, which is what the network needs to learn a smooth change with speed.
    s.reference = readmatrix(fullfile(here, '..', 'outputs', 'part1', 'tournament_twopoint_seed1', 'best_chromosome.csv'));
    s.range = high - low;
    low([3 4]) = s.reference([3 4]);
    high([3 4]) = s.reference([3 4]);
    outName = sprintf('speed_%.2f', targetSpeed);
end
outDir = fullfile(here, '..', 'outputs', 'part1', outName);
if ~exist(outDir, 'dir'), mkdir(outDir); end

%% The GA
rng(seed);
pop = low + rand(popSize, nGenes) .* (high - low);   % random starting population
if ~isnan(targetSpeed)                               % ...or one around the reference gait
    pop = min(max(s.reference + 0.1 * randn(popSize, nGenes) .* (high - low), low), high);
    pop(1, :) = min(max(s.reference, low), high);
end
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
[gait, down, feet] = decode_gait(best, s.nFrames);
P = leg_fk(gait);
fprintf('best fitness %.3f\n', score);
fprintf('planned speed %.3f m/s, out of reach %.4f, past limits %.4f, too fast %.3f, legs too close %.4f, unstable %.1f%%\n', ...
    info.speed, info.outOfReach, info.pastLimits, info.tooFast, info.legsTooClose, 100 * info.unstable);
names = {'stride', 'stepHeight', 'duty', 'period', 'bodyHeight', 'reach', 'spread', 'xShift'};
for k = 1:nGenes
    fprintf('  %-10s %.3f\n', names{k}, best(k));
end

% check leg_ik.m against leg_fk.m: the feet must end up where we asked
err = max(vecnorm(reshape(P(:, :, 4, :), [], 3) - reshape(feet, [], 3), 2, 2));
fprintf('biggest IK error: %.1e m\n', err);

% files for the Webots controller (controllers/gait_player)
writematrix(gait, fullfile(outDir, 'joint_table.csv'));      % 300 x 24 motor angles, one pose per row
writematrix(double(down), fullfile(outDir, 'stance.csv'));   % 300 x 8, 1 = foot on the ground
standFeet = feet(1, :, :);
standFeet(1, :, 1) = best(7) * [0.4875 0.1625 -0.1625 -0.4875 -0.4875 -0.1625 0.1625 0.4875] + best(8);
standFeet(1, :, 3) = -best(5);
writematrix(leg_ik(standFeet), fullfile(outDir, 'stand.csv'));   % standing pose before walking
meta = table(["frames"; "period"; "stride"; "duty"; "stepHeight"; "bodyHeight"; "speed"], ...
    [s.nFrames; best(4); best(1); best(3); best(2); best(5); info.speed], 'VariableNames', {'key', 'value'});
writetable(meta, fullfile(outDir, 'meta.csv'));
writematrix(best, fullfile(outDir, 'best_chromosome.csv'));
writematrix([(1:numel(bestHist))', bestHist', meanHist'], fullfile(outDir, 'fitness_history.csv'));

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
plot(P(:, :, 4, 3) + best(5)); grid on;
xlabel('Frame'); ylabel('Height above ground');
legend(names, 'Location', 'eastoutside');
title('Foot heights over one cycle');
subplot(2, 1, 2);
imagesc(~down'); colormap(gray);
yticks(1:8); yticklabels(names); xlabel('Frame');
title('Feet on the ground (black)');
saveas(figure(3), fullfile(outDir, 'feet.png'));

% first pose and an animation of the robot (legs drawn from leg_fk.m)
gifFile = fullfile(outDir, 'gait.gif');
figure(1); lightTheme(1);
for f = 1:5:s.nFrames
    drawRobot(squeeze(P(f, :, :, :)), down(f, :), best(5));
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


function drawRobot(P, down, bodyHeight)
% P is 8 x 4 x 3 (leg, point, xyz). Feet on the ground are drawn red.
clf; hold on; grid on; axis equal;
fill3([0.65 0.65 -0.65 -0.65], [0.2 -0.2 -0.2 0.2], [0 0 0 0], [0.6 0.6 0.6]);   % body (seen from above)
for leg = 1:8
    plot3(P(leg, :, 1), P(leg, :, 2), P(leg, :, 3), 'b-', 'LineWidth', 2);
    colour = 'b';
    if down(leg), colour = 'r'; end
    plot3(P(leg, 4, 1), P(leg, 4, 2), P(leg, 4, 3), 'o', 'MarkerFaceColor', colour, 'MarkerEdgeColor', colour);
end
plot3(0.8, 0, 0, 'k^', 'MarkerFaceColor', 'k');       % front
patch([-2 2 2 -2], [-1.6 -1.6 1.6 1.6], -bodyHeight * [1 1 1 1], [0.85 0.85 0.85], 'FaceAlpha', 0.4);  % ground
xlim([-2 2]); ylim([-1.6 1.6]); zlim([-bodyHeight - 0.05, 0.9]);
xlabel('x (forward)'); ylabel('y'); zlabel('z'); view(35, 25);
hold off;
end


function lightTheme(n)
% newer MATLAB versions draw dark figures by default; older ones have no theme()
try
    theme(figure(n), 'light');
catch
end
end
