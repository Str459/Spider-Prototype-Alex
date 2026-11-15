function [q, excess] = robot_ik(T, leg, g)
%ROBOT_IK Exact inverse kinematics of Webots SpiderOctopod legs (vectorised).
%   [q, excess] = ROBOT_IK(T, leg, g)
%     T    N x 3 wanted foot positions in the body frame
%     leg  N x 1 leg index (robot_geometry order)
%   returns
%     q       N x 3 joint angles [hip thigh knee] (knee-up solution)
%     excess  N x 1 how far (m) the target is outside the leg's reach; 0 if
%             reachable. Joint limits are NOT applied here: the fitness
%             function measures and penalises any violation instead.
%
%   1. Hip: choose the hip angle that turns the leg's plane through the
%      target (the plane must contain the foot).
%   2. Thigh and knee: a two-link planar problem in that plane, solved with
%      the cosine rule, taking the "knee up" branch like a real spider.

leg = leg(:);
N = size(T, 1);
s = g.side(leg)';
A = [g.hipX(leg)', g.hipY(leg)', zeros(N, 1)];
a = [zeros(N, 1), -s / sqrt(2), ones(N, 1) / sqrt(2)];   % hip axis
d = T - A;

% Hip angle q1 so that rotating d back by q1 gives x = 0:
%   d_x cos(q1) - (a x d)_x sin(q1) = 0
cx = (-s .* d(:, 3) - d(:, 2)) / sqrt(2);
q1 = atan2(d(:, 1) .* sign(cx), abs(cx));
w  = rodrigues(d, a, -q1);
u  = s .* w(:, 2);                  % outwards in the leg plane
z  = w(:, 3);

c  = g.coxa * cos(g.coxaElev);
du = u - c;
dz = z - c;
D  = hypot(du, dz);
Lmax = g.thigh + g.tibia - 1e-9;
Lmin = abs(g.tibia - g.thigh) + 1e-9;
excess = max(0, D - Lmax) + max(0, Lmin - D);
Dc = min(max(D, Lmin), Lmax);

gam = acos(min(max((g.thigh^2 + Dc.^2 - g.tibia^2) ./ (2 * g.thigh * Dc), -1), 1));
t2 = atan2(dz, du) + gam;                                  % knee up
t3 = atan2(dz - g.thigh * sin(t2), du - g.thigh * cos(t2));
q2 = s .* (t2 - pi / 4);
q3 = s .* (t3 + pi / 4) - q2;
q = [q1, q2, q3];
end

function v = rodrigues(v, a, th)
c = cos(th);
s = sin(th);
v = v .* c + cross(a, v, 2) .* s + a .* sum(a .* v, 2) .* (1 - c);
end
