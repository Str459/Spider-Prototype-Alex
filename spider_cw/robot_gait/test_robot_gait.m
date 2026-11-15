function test_robot_gait()
%TEST_ROBOT_GAIT Self-checks for the Webots-robot gait model.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'part1_ga'));
g = robot_geometry();
L = @(name) find(strcmp(g.labels, name));

% 1. FK matches the controller's step-by-step proto chain (values printed by
%    the Python robot_foot() of the backed-up controller).
ref = {'R1', [0 1.07 -0.58],  [0.4875, -1.333202058, -0.4845922187]
       'L3', [0.3 -0.8 0.2],  [-0.3169903675, 1.2660157588, -0.3597216429]
       'R4', [-0.4 0.5 -1.2], [-1.0628663477, -1.7442605684, 0.3802993731]
       'L1', [0.7 0.2 -1.5],  [0.2210842799, 0.4717747018, 0.1755408891]};
err = 0;
for k = 1:size(ref, 1)
    J = robot_fk(ref{k, 2}, L(ref{k, 1}), g);
    err = max(err, max(abs(squeeze(J(1, 4, :))' - ref{k, 3})));
end
fprintf('[1] robot_fk vs controller chain: max error %.2g m\n', err);
assert(err < 1e-8);

% 2. IK -> FK round trip on random knee-up poses inside the limits.
rng(2);
N = 5000;
leg = randi(8, N, 1);
s = g.side(leg)';
q = [(rand(N, 1) - 0.5) * 2.0, s .* (rand(N, 1) * 1.6 - 1.0), s .* (-rand(N, 1) * 1.4)];
J = robot_fk(q, leg, g);
Tg = squeeze(J(:, 4, :));
[q2, ex] = robot_ik(Tg, leg, g);
J2 = robot_fk(q2, leg, g);
rt = max(vecnorm(squeeze(J2(:, 4, :)) - Tg, 2, 2));
fprintf('[2] IK round trip on %d poses: max foot error %.2g m, max reach excess %.2g\n', N, rt, max(ex));
assert(rt < 1e-8 && max(ex) == 0);

% 3. Segment distance.
d = seg_dist([0 0 0], [1 0 0], [0.5 1 0], [0.5 2 0]);
d2 = seg_dist([0 0 0], [1 0 0], [2 0 1], [3 0 1]);
fprintf('[3] seg_dist: %.3f (expect 1), %.3f (expect %.3f)\n', d, d2, sqrt(2));
assert(abs(d - 1) < 1e-12 && abs(d2 - sqrt(2)) < 1e-12);

% 4. Groups: phaseB = 0.5, wave = 0 gives exactly the two tetrapod groups.
genes = robot_gait_genes();
c = (genes.lb + genes.ub) / 2;
c(8) = 0.5;
c(9) = 0;
[~, st] = robot_gait_decode(c, 100, g);
st = squeeze(st);
A = [L('L1') L('L3') L('R2') L('R4')];
B = [L('L2') L('L4') L('R1') L('R3')];
sameA = all(all(st(:, A) == st(:, A(1))));
sameB = all(all(st(:, B) == st(:, B(1))));
overlap = mean(all(st, 2));
fprintf('[4] group A in step: %d, group B in step: %d, all 8 feet down %.0f%% of the cycle\n', ...
    sameA, sameB, 100 * overlap);
assert(sameA && sameB && overlap > 0);

% 5. Standing still never beats walking.
fcfg = robot_fitness_config();
still = c;
still(1) = genes.lb(1) * 0;           % stride 0
[sStill, cStill] = robot_gait_fitness(still, fcfg, g);
[sWalk, cWalk] = robot_gait_fitness(c, fcfg, g);
fprintf('[5] stride 0: score %.3f (speed %.3f); mid-range walk: score %.3f (speed %.3f)\n', ...
    sStill, cStill.speed, sWalk, cWalk.speed);
assert(sStill < 0);
fprintf('ALL ROBOT GAIT SELF-CHECKS PASSED\n');
end
