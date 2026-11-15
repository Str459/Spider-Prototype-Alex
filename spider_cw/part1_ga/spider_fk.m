function [feet, joints] = spider_fk(angles, geom)
%SPIDER_FK Vectorised forward kinematics of the lecturer's spider model.
%   feet = SPIDER_FK(angles) takes an N x 24 matrix (one pose per row, leg
%   order [L1 L2 L3 L4 R4 R3 R2 R1], joints [coxa femur tibia] in radians)
%   and returns an N x 8 x 3 array of foot (tibia tip) positions.
%
%   [feet, joints] = SPIDER_FK(...) also returns N x 8 x 4 x 3 joint
%   positions (body attachment, end of coxa, end of femur, foot) for drawing.
%
%   Why not just call the provided forward_leg_kinematics2.m?
%   One GA run scores millions of leg poses. The provided function builds
%   three rotation matrices per leg and can only do one leg at a time, which
%   would make a run take hours. Working the provided code through by hand
%   (Rodrigues' rotation formula) shows that each leg stays in a vertical
%   plane with heading psi = baseAngle + theta1, and that:
%       coxa  pitch      = 30 deg (fixed)
%       femur pitch  P   = 30 deg + theta2
%       tibia pitch  T   = P + sign(cos P) * theta3
%   The sign(cos P) term reproduces a quirk of the provided code: it
%   re-normalises cross(femur_dir, z), and that axis flips over when the
%   femur points backwards past vertical. test_part1.m checks this function
%   against forward_leg_kinematics2.m on random poses (error ~1e-15).

if nargin < 2
    geom = spider_geometry();
end
N  = size(angles, 1);
t1 = angles(:, 1:3:end);    % N x 8 coxa yaw
t2 = angles(:, 2:3:end);    % N x 8 femur pitch
t3 = angles(:, 3:3:end);    % N x 8 tibia pitch

psi = geom.baseAngle + t1;  % heading of each leg's vertical plane
c = cos(psi);
s = sin(psi);
E = geom.coxaElev;
P = E + t2;
T = P + sign(cos(P)) .* t3;
L = geom.segLen;

bx = repmat(geom.basePos(:, 1)', N, 1);
by = repmat(geom.basePos(:, 2)', N, 1);
bz = zeros(N, 8);

% Horizontal reach (h) and height change (v) of each segment.
h1 = L(1) * cos(E);   v1 = L(1) * sin(E);
h2 = L(2) * cos(P);   v2 = L(2) * sin(P);
h3 = L(3) * cos(T);   v3 = L(3) * sin(T);

j2 = cat(3, bx + h1 .* c,         by + h1 .* s,         bz + v1);
j3 = cat(3, j2(:,:,1) + h2 .* c,  j2(:,:,2) + h2 .* s,  j2(:,:,3) + v2);
j4 = cat(3, j3(:,:,1) + h3 .* c,  j3(:,:,2) + h3 .* s,  j3(:,:,3) + v3);
feet = j4;

if nargout > 1
    j1 = cat(3, bx, by, bz);
    joints = permute(cat(4, j1, j2, j3, j4), [1 2 4 3]);   % N x 8 x 4 x 3
end
end
