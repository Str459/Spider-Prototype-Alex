function [J, kHip, kKnee] = robot_fk(q, leg, g)
%ROBOT_FK Forward kinematics of Webots SpiderOctopod legs (vectorised).
%   [J, kHip, kKnee] = ROBOT_FK(q, leg, g)
%     q    N x 3 joint angles [hip thigh knee] (radians, Webots motor values)
%     leg  N x 1 leg index into robot_geometry order [L1..L4 R4..R1]
%   returns
%     J     N x 4 x 3 points: hip anchor, thigh joint, knee, foot (body frame)
%     kHip  N x 3 hip axis, kKnee N x 3 thigh/knee axis (for torque estimates)
%
%   Working through the proto: before the hip turns, every leg lies in the
%   plane x = 0. In that plane (u = outwards, z = up), relative to the hip:
%     coxa  0.5 m at +45 deg
%     thigh 0.6 m at  45 deg + s*q_thigh
%     tibia 0.7 m at -45 deg + s*(q_thigh + q_knee)      (s = +1 left, -1 right)
%   The hip then turns that plane about the axis (0, -s, 1)/sqrt(2).
%   test_robot_gait.m checks this against the controller's step-by-step
%   implementation of the proto's joint chain.

leg = leg(:);
N = size(q, 1);
s  = g.side(leg)';
c  = g.coxa * cos(g.coxaElev);
t2 = pi / 4 + s .* q(:, 2);
t3 = -pi / 4 + s .* (q(:, 2) + q(:, 3));
u2 = c * ones(N, 1);            z2 = c * ones(N, 1);
u3 = u2 + g.thigh * cos(t2);    z3 = z2 + g.thigh * sin(t2);
u4 = u3 + g.tibia * cos(t3);    z4 = z3 + g.tibia * sin(t3);

kHip = [zeros(N, 1), -s / sqrt(2), ones(N, 1) / sqrt(2)];
A  = [g.hipX(leg)', g.hipY(leg)', zeros(N, 1)];
z0 = zeros(N, 1);
P2 = A + rodrigues([z0, s .* u2, z2], kHip, q(:, 1));
P3 = A + rodrigues([z0, s .* u3, z3], kHip, q(:, 1));
P4 = A + rodrigues([z0, s .* u4, z4], kHip, q(:, 1));
J = permute(cat(3, A, P2, P3, P4), [1 3 2]);           % N x 4 x 3
kKnee = rodrigues(repmat([1 0 0], N, 1), kHip, q(:, 1));
end

function v = rodrigues(v, a, th)
% Rotate rows of v about unit axes a by angles th.
c = cos(th);
s = sin(th);
v = v .* c + cross(a, v, 2) .* s + a .* sum(a .* v, 2) .* (1 - c);
end
