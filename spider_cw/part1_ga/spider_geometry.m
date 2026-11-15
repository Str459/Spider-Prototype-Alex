function geom = spider_geometry()
%SPIDER_GEOMETRY Constants of the lecturer's 3D spider model.
%   All values are copied from plot_spider_pose.m and
%   forward_leg_kinematics2.m (in the "AI cw" folder), so that our
%   vectorised kinematics (spider_fk.m) reproduces the provided model
%   exactly. Units are the model's own length units.
%
%   Leg order everywhere in this project follows the coursework spec:
%       [L1 L2 L3 L4 R4 R3 R2 R1]  (front-left, round the back, to front-right)
%   so one pose is the 1x24 vector [L1a L1b L1c ... R1a R1b R1c] in radians,
%   where a = coxa (yaw), b = femur (pitch), c = tibia (pitch).
%
%   Body frame: +x is forward (the head), +y is left, +z is up.

geom.nLegs     = 8;
geom.labels    = {'L1','L2','L3','L4','R4','R3','R2','R1'};
geom.segLen    = [1.2, 0.7, 1.0];       % [coxa femur tibia] lengths
geom.bodyA     = 1.5;                   % body ellipse semi-axis along x
geom.bodyB     = 1.0;                   % body ellipse semi-axis along y
geom.baseAngle = deg2rad([45 75 105 135 -135 -105 -75 -45]);
geom.coxaElev  = deg2rad(30);           % fixed upward pitch of every coxa

% Where each leg is attached to the body ellipse (8 x 3).
geom.basePos = [geom.bodyA * cos(geom.baseAngle); ...
                geom.bodyB * sin(geom.baseAngle); ...
                zeros(1, 8)]';

geom.isLeft = [true true true true false false false false];

% Front-to-back pair index: 1 = front (L1/R1) ... 4 = rear (L4/R4).
% The gait encoding gives the left and right leg of a pair the same genes.
geom.pair = [1 2 3 4 4 3 2 1];

% Increasing theta1 turns a leg anticlockwise seen from above. That moves a
% LEFT foot backwards and a RIGHT foot forwards. Multiplying a "forward"
% coxa angle by this sign gives the model's raw theta1, so the same gene
% value means the same thing on both sides of the body.
geom.fwdSign = [-1 -1 -1 -1 1 1 1 1];

% Feet that sit next to each other going round the body (could collide).
geom.neighbours = [1 2; 2 3; 3 4; 4 5; 5 6; 6 7; 7 8; 8 1];
end
