function [score, comp] = pose_fitness(pop, fcfg, geom)
%POSE_FITNESS Score single standing poses (Stage A). Higher is better.
%   pop is P x 24: the chromosome IS the 1x24 joint-angle vector of the
%   spec, [L1a L1b L1c ... R1a R1b R1c], in radians.
%
%   A good standing pose has:
%     all 8 feet on the ground           (groundErr: mean |foot z - ground|)
%     no foot below the ground           (penetration)
%     a wide support polygon             (margin, rewarded up to 1.5)
%     left/right mirror symmetry         (symmetry: mean angle mismatch)
%     neighbouring feet apart            (collision, as in the gait)
%   Weights are chosen so that a centimetre-scale ground error costs about
%   as much as a large drop in margin: being on the ground comes first.

feet = spider_fk(pop, geom);
X = feet(:, :, 1);
Y = feet(:, :, 2);
Z = feet(:, :, 3);
dz = Z - fcfg.groundZ;

groundErr = mean(abs(dz), 2);
penetr    = mean(max(0, -dz), 2);
margin    = support_margin(X, Y, true(size(X)));

% Mirror partner of L1..L4 is R1..R4, which sit at columns 8,7,6,5.
L = 1:4;
R = [8 7 6 5];
t1 = pop(:, 1:3:end);
t2 = pop(:, 2:3:end);
t3 = pop(:, 3:3:end);
symmetry = mean([abs(t1(:, L) + t1(:, R)), abs(t2(:, L) - t2(:, R)), abs(t3(:, L) - t3(:, R))], 2);

collision = zeros(size(pop, 1), 1);
for k = 1:size(geom.neighbours, 1)
    a = geom.neighbours(k, 1);
    b = geom.neighbours(k, 2);
    gap = hypot(X(:, a) - X(:, b), Y(:, a) - Y(:, b));
    collision = collision + max(0, fcfg.footGap - gap);
end

score = 2 * min(margin, 1.5) - 20 * groundErr - 30 * penetr - 2 * symmetry - 20 * collision;
comp = struct('margin', margin, 'groundErr', groundErr, 'penetration', penetr, ...
    'symmetry', symmetry, 'collision', collision, 'score', score);
end
