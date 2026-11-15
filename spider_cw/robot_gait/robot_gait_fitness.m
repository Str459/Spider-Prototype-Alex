function [score, comp, detail] = robot_gait_fitness(chrom, fcfg, g)
%ROBOT_GAIT_FITNESS Score gait genes for the Webots robot (higher is better).
%   [score, comp, detail] = ROBOT_GAIT_FITNESS(chrom, fcfg, g)
%   fcfg comes from robot_fitness_config.m.
%
%   GOAL: forward body speed (m/s). Planted feet move straight back at
%   stride / (duty * period), so that is how fast the body travels.
%   A gait slower than fcfg.minSpeed also pays a fixed "not walking"
%   penalty, so standing still (or shuffling) can never win.
%
%   REQUIREMENTS (cost nothing when met, checked on every frame):
%     reach        every foot target reachable by its leg
%     limits       every joint inside its limits with a safety margin
%     joint speed  below fcfg.maxVel (80% of the 2 rad/s motor limit)
%     torque       static joint torque below fcfg.maxTorque (the 43 kg robot
%                  shared between the planted feet; motors give 50 Nm)
%     collisions   coxa/thigh/tibia of neighbouring legs at least
%                  fcfg.minClear apart (self-collision is on in Webots)
%     support      at least 4 feet down in every frame
%     stability    body centre at least fcfg.minMargin inside the polygon
%                  of planted feet
%   SMOOTHNESS: a small cost on joint acceleration (no jerky moves).

P = size(chrom, 1);
F = fcfg.nFrames;
[T, stance, period] = robot_gait_decode(chrom, F, g);
legIdx = repmat(reshape(1:8, 1, 1, 8), P, F, 1);
[q, excess] = robot_ik(reshape(T, P * F * 8, 3), legIdx(:), g);
[J, kHip, kKnee] = robot_fk(q, legIdx(:), g);
q = reshape(q, P, F, 8, 3);
J = reshape(J, P, F, 8, 4, 3);
kHip = reshape(kHip, P, F, 8, 3);
kKnee = reshape(kKnee, P, F, 8, 3);

% Reach and joint limits ---------------------------------------------------
reachErr = max(reshape(excess, P, []), [], 2);
lo = reshape(g.qMin, 1, 1, 1, 3) + fcfg.limitMargin;
hi = reshape(g.qMax, 1, 1, 1, 3) - fcfg.limitMargin;
limitErr = max(reshape(max(0, lo - q) + max(0, q - hi), P, []), [], 2);

% Joint speed and acceleration (the gait loops, so frame F joins frame 1) --
rate = F ./ period;                                          % frames per second
dq  = (circshift(q, -1, 2) - q) .* rate;
ddq = (circshift(q, -1, 2) - 2 * q + circshift(q, 1, 2)) .* rate .^ 2;
maxJointVel = max(reshape(abs(dq), P, []), [], 2);
velErr = max(0, maxJointVel - fcfg.maxVel);
accRms = sqrt(mean(reshape(ddq .^ 2, P, []), 2));

% Clearance between neighbouring legs --------------------------------------
minClear = inf(P, 1);
for k = 1:size(g.neighbours, 1)
    a = g.neighbours(k, 1);
    b = g.neighbours(k, 2);
    for i = 1:3
        for j = 1:3
            d = seg_dist(seg(J, a, i), seg(J, a, i + 1), seg(J, b, j), seg(J, b, j + 1));
            clear = reshape(d, P, F) - g.radius(i) - g.radius(j);
            minClear = min(minClear, min(clear, [], 2));
        end
    end
end
colErr = max(0, fcfg.minClear - minClear);

% Support and static stability ----------------------------------------------
fx = reshape(J(:, :, :, 4, 1), P * F, 8);
fy = reshape(J(:, :, :, 4, 2), P * F, 8);
[margin, nDown] = support_margin(fx, fy, reshape(stance, P * F, 8));
margin = reshape(margin, P, F);
nDown = reshape(nDown, P, F);
minMargin = min(margin, [], 2);
minFeet = min(nDown, [], 2);
stabErr = max(0, fcfg.minMargin - minMargin);
supportErr = mean(nDown < 4, 2);

% Static torque on planted legs --------------------------------------------
load = g.mass * 9.81 ./ max(nDown, 1);                       % N per planted foot
foot = J(:, :, :, 4, :);
tq = zeros(P, F, 8, 3);
pivots = {J(:, :, :, 1, :), J(:, :, :, 2, :), J(:, :, :, 3, :)};
axes = {kHip, kKnee, kKnee};
for j = 1:3
    rr = squeeze5(foot - pivots{j});                         % P x F x 8 x 3
    k = axes{j};
    % torque of an upward force about axis k: F * (k_x r_y - k_y r_x)
    tq(:, :, :, j) = abs(k(:, :, :, 1) .* rr(:, :, :, 2) - k(:, :, :, 2) .* rr(:, :, :, 1));
end
tq = tq .* load .* stance;
maxTorque = max(reshape(tq, P, []), [], 2);
torqueErr = max(0, maxTorque - fcfg.maxTorque);

% Goal ------------------------------------------------------------------------
speed = chrom(:, 1) ./ (chrom(:, 3) .* period);              % m/s
notWalking = double(speed < fcfg.minSpeed);
% Foot speed over the ground at touch-down / lift-off. A planted foot must
% have zero ground speed; a foot that lands with ground speed skids.
contactSlip = (1 - chrom(:, 11)) .* speed;                   % m/s

w = fcfg.w;
score = w.speed * speed ...
      - w.reach * reachErr - w.limit * limitErr - w.vel * velErr ...
      - w.torque * torqueErr - w.collision * colErr - w.support * supportErr ...
      - w.stability * stabErr - w.acc * accRms - w.still * notWalking ...
      - w.contactSlip * contactSlip;

comp = struct('speed', speed, 'contactSlip', contactSlip, 'reachErr', reachErr, 'limitErr', limitErr, ...
    'maxJointVel', maxJointVel, 'maxTorque', maxTorque, 'minClear', minClear, ...
    'minMargin', minMargin, 'minFeet', minFeet, 'accRms', accRms, 'score', score);
if nargout > 2
    detail = struct('T', T, 'stance', stance, 'q', q, 'J', J, 'period', period, ...
        'margin', margin, 'nDown', nDown, 'jointVel', dq, 'torque', tq);
end
end

function p = seg(J, leg, k)
p = reshape(J(:, :, leg, k, :), [], 3);
end

function x = squeeze5(x)
s = size(x);
x = reshape(x, s(1), s(2), s(3), 3);
end
