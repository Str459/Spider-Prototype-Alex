function [q, outOfReach] = leg_ik(feet)
% Inverse kinematics of the Webots spider: where each foot should be -> motor angles.
%   feet        n x 8 x 3 foot targets (body frame, metres), legs L1 L2 L3 L4 R4 R3 R2 R1
%   q           n x 24 motor angles (hip, thigh, knee per leg)
%   outOfReach  n x 8, how far (m) a target is outside what the leg can reach (0 = fine)
% leg_fk.m does the opposite; run_ga.m checks that the two agree.
%
% Step 1: the hip turns the leg's plane, so pick the hip angle that turns the
%         plane through the target.
% Step 2: thigh + tibia is then a 2-link arm in that plane, solved with the
%         cosine rule. We take the "knee up" answer, like a real spider.

hipX = [0.4875 0.1625 -0.1625 -0.4875 -0.4875 -0.1625 0.1625 0.4875];
side = [1 1 1 1 -1 -1 -1 -1];
n = size(feet, 1);
q = zeros(n, 24);
outOfReach = zeros(n, 8);
c = 0.5 * cos(pi / 4);

for leg = 1:8
    s = side(leg);
    d = reshape(feet(:, leg, :), n, 3) - [hipX(leg), 0.2 * s, 0];   % target relative to the hip
    axis = [0, -s, 1] / sqrt(2);

    % step 1: hip angle (turning the target back by q1 must put it in the plane x = 0)
    cx = (-s * d(:, 3) - d(:, 2)) / sqrt(2);
    q1 = atan2(d(:, 1) .* sign(cx), abs(cx));
    w = d .* cos(-q1) + cross(repmat(axis, n, 1), d, 2) .* sin(-q1) + axis .* (d * axis') .* (1 - cos(-q1));
    du = s * w(:, 2) - c;                     % target seen from the thigh joint, in the leg plane
    dz = w(:, 3) - c;

    % step 2: two-link arm (thigh 0.6, tibia 0.7)
    D = hypot(du, dz);
    outOfReach(:, leg) = max(0, D - 1.3) + max(0, 0.1 - D);
    D = min(max(D, 0.1 + 1e-9), 1.3 - 1e-9);
    bend = acos((0.6^2 + D.^2 - 0.7^2) ./ (2 * 0.6 * D));
    a2 = atan2(dz, du) + bend;                % thigh direction (knee up)
    a3 = atan2(dz - 0.6 * sin(a2), du - 0.6 * cos(a2));   % tibia direction
    q2 = s * (a2 - pi / 4);
    q3 = s * (a3 + pi / 4) - q2;
    q(:, 3 * leg - 2:3 * leg) = [q1, q2, q3];
end
end
