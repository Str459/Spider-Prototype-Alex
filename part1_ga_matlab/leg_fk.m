function P = leg_fk(q)
% Forward kinematics of the Webots spider (protos/SpiderOctopod.proto).
%   q  n x 24 motor angles, legs L1 L2 L3 L4 R4 R3 R2 R1, each (hip, thigh, knee)
%   P  n x 8 x 4 x 3 points of every leg: hip, thigh joint, knee, foot
%      (body frame in metres: x forward, y left, z up, origin = body centre)
%
% Each leg is a coxa (0.5 m, tilted 45 degrees up), thigh (0.6 m) and tibia (0.7 m).
% Thigh and knee bend the leg inside a vertical plane. The hip turns that whole
% plane about an axis tilted 45 degrees. Left and right legs are mirror images
% (s = +1 left, -1 right).

hipX = [0.4875 0.1625 -0.1625 -0.4875 -0.4875 -0.1625 0.1625 0.4875];
side = [1 1 1 1 -1 -1 -1 -1];
n = size(q, 1);
P = zeros(n, 8, 4, 3);
c = 0.5 * cos(pi / 4);                         % coxa end, outwards and up

for leg = 1:8
    s = side(leg);
    q1 = q(:, 3 * leg - 2);
    q2 = q(:, 3 * leg - 1);
    q3 = q(:, 3 * leg);
    % points in the leg's own plane: u = outwards, z = up (relative to the hip)
    a2 = pi / 4 + s * q2;                      % thigh direction
    a3 = -pi / 4 + s * (q2 + q3);              % tibia direction
    u = [zeros(n, 1), c + 0 * q2, c + 0.6 * cos(a2), c + 0.6 * cos(a2) + 0.7 * cos(a3)];
    z = [zeros(n, 1), c + 0 * q2, c + 0.6 * sin(a2), c + 0.6 * sin(a2) + 0.7 * sin(a3)];
    axis = [0, -s, 1] / sqrt(2);               % hip axis
    for k = 1:4
        v = [zeros(n, 1), s * u(:, k), z(:, k)];
        v = rotate(v, axis, q1);
        P(:, leg, k, :) = reshape(v + [hipX(leg), 0.2 * s, 0], n, 1, 1, 3);
    end
end
end


function v = rotate(v, a, angle)
% Rodrigues' formula: rotate each row of v about the unit axis a
v = v .* cos(angle) + cross(repmat(a, size(v, 1), 1), v, 2) .* sin(angle) ...
    + a .* (v * a') .* (1 - cos(angle));
end
