function test_part1()
%TEST_PART1 Self-checks for Part 1. Run from MATLAB:  test_part1
%   1. spider_fk.m matches the lecturer's forward_leg_kinematics2.m
%   2. support_margin.m gives the right answer on simple shapes
%   3. gait_decode.m output sizes, looping and planted-foot height
%   4. the fitness prefers a hand-built walk over marching on the spot
%   5. the GA engine solves a standard test function (sphere)
%   The hand-built gait is ONLY used here, as a sanity check of the fitness
%   function. It is never put into the GA's population.

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', '..', 'AI cw'));     % lecturer's files
geom = spider_geometry();

% 1 ---------------------------------------------------------------------
rng(1);
N = 200;
angles = (rand(N, 24) - 0.5) * 2 * pi;            % any angles, even extreme ones
angles(1, :) = 0;
[~, joints] = spider_fk(angles, geom);
maxErr = 0;
for i = 1:N
    for leg = 1:8
        th = angles(i, 3 * leg - 2 : 3 * leg);
        [j1, j2, j3, j4] = forward_leg_kinematics2(geom.basePos(leg, :), ...
            geom.baseAngle(leg), th, geom.segLen);
        ref  = [j1; j2; j3; j4];
        mine = squeeze(joints(i, leg, :, :));
        maxErr = max(maxErr, max(abs(ref(:) - mine(:))));
    end
end
fprintf('[1] spider_fk vs forward_leg_kinematics2 (%d poses x 8 legs): max error %.2g\n', N, maxErr);
assert(maxErr < 1e-9, 'spider_fk does not match the provided model');

% 2 ---------------------------------------------------------------------
sq = [1 1; -1 1; -1 -1; 1 -1];                    % square, centre inside, distance 1
x = [sq(:, 1)', 5 5 5 5];
y = [sq(:, 2)', 5 5 5 5];
m1 = support_margin(x, y, logical([1 1 1 1 0 0 0 0]));
m2 = support_margin(x, y, logical([1 1 0 0 0 0 0 0]));            % only 2 feet
m3 = support_margin([1 2 1 0 0 0 0 0], [1 0 -1 0 0 0 0 0], logical([1 1 1 0 0 0 0 0])); % centre outside
fprintf('[2] support_margin: square %.3f (expect 1), two feet %.1f (expect -1), outside %.3f (expect < 0)\n', m1, m2, m3);
assert(abs(m1 - 1) < 1e-12 && m2 == -1 && m3 < 0, 'support_margin wrong');

% 3 ---------------------------------------------------------------------
walk = hand_gait(0.25);
[a, st] = gait_decode(walk, 300, geom);
assert(isequal(size(a), [1 300 24]) && isequal(size(st), [1 300 8]), 'decode sizes wrong');
loopJump = max(abs(a(1, 1, :) - a(1, end, :)));
fcfg = fitness_config();
fcfg.nFrames = 300;
[~, ~, det] = gait_fitness(walk, fcfg, geom);
zPlanted = det.feet(:, :, :, 3);
zPlanted = zPlanted(det.stance);
fprintf('[3] decode 1x300x24 ok; frame 300 -> 1 jump %.3f rad; planted feet z in [%.4f, %.4f] (ground %.2f)\n', ...
    loopJump, min(zPlanted), max(zPlanted), fcfg.groundZ);
assert(loopJump < 0.05, 'gait does not loop');
assert(max(abs(zPlanted - fcfg.groundZ)) < 1e-3, 'planted feet leave the ground');

% 4 ---------------------------------------------------------------------
fcfg = fitness_config();
[sWalk, cWalk] = gait_fitness(walk, fcfg, geom);
[sStill, cStill] = gait_fitness(hand_gait(0), fcfg, geom);
fprintf('[4] hand-built walk: score %.3f, progress %.3f, min margin %.3f, frames short of feet %.1f%%\n', ...
    sWalk, cWalk.progress, cWalk.minMargin, 100 * cWalk.stanceDeficit);
fprintf('    marching on the spot: score %.3f, progress %.3f\n', sStill, cStill.progress);
assert(sWalk > sStill, 'fitness does not reward walking');

% 5 ---------------------------------------------------------------------
prob.nGenes = 10;
prob.lb = -5 * ones(1, 10);
prob.ub =  5 * ones(1, 10);
prob.circular = false(1, 10);
prob.evaluate = @sphere;
cfg = ga_default_config();
cfg.verbose = false;
cfg.maxGen = 200;
names = {'tournament', 'roulette', 'rank'};
for k = 1:3
    cfg.selection = names{k};
    r = ga_run(prob, cfg);
    fprintf('[5] sphere test, %-10s selection: best %.2e after %d generations\n', ...
        names{k}, -r.bestScore, r.generations);
    % Random start is about 83. Roulette is expected to be the weakest:
    % once scores are close its selection pressure almost disappears.
    assert(-r.bestScore < 1, 'GA failed on the sphere function');
end
fprintf('ALL PART 1 SELF-CHECKS PASSED\n');
end

function chrom = hand_gait(stride)
% A walking gait built by hand: alternating tetrapod, front legs reaching
% forward, rear legs back. Femur/tibia chosen so planted feet sit at z = -0.4.
% stride = coxa swing amplitude (0 = marching on the spot).
offs = [0.25 0.08 -0.08 -0.25];
chrom = zeros(1, 27);
for k = 1:4
    chrom((k - 1) * 6 + (1:6)) = [offs(k), stride, -0.6981, 0.5, -0.8971, 0];
end
chrom(25:27) = [0.5 0.5 0.6];
end

function [s, c] = sphere(pop)
s = -sum(pop .^ 2, 2);
c = struct('value', -s);
end
