function g = robot_geometry()
%ROBOT_GEOMETRY Facts about the Webots SpiderOctopod robot, read from
%   protos/SpiderOctopod.proto and worlds/spider_gait.wbt.
%
%   Frame: body centre, +x forward (legs 1 are at the front), +y left, +z up.
%   Leg order matches Part 1 and the CSVs: [L1 L2 L3 L4 R4 R3 R2 R1].
%   Each leg: hip (about an axis tilted 45 deg, perpendicular to the coxa),
%   thigh and knee (both about the coxa's x axis).
%   Coxa 0.5 m pointing out and 45 deg up, thigh 0.6 m, tibia 0.7 m.
%   Limits: hip and thigh +-1.5708 rad, knee [-2, 1] rad on EVERY leg. The
%   left and right joints are mirror images, so the knee can bend 2 rad one
%   way on the right but only 1 rad on the left: a symmetric gait must keep
%   the knee bend within +-1 rad.
%   Motors: max speed 2 rad/s, max torque 50 Nm. Basic time step 32 ms.
%   Collision shapes: body box 1.3 x 0.4 x 0.3, coxa cylinder r 0.03,
%   thigh box 0.04 x 0.04 (half diagonal 0.028), tibia cylinder r 0.02.

g.labels = {'L1','L2','L3','L4','R4','R3','R2','R1'};
g.side   = [1 1 1 1 -1 -1 -1 -1];            % +1 left, -1 right
g.pair   = [1 2 3 4 4 3 2 1];                % 1 = front ... 4 = rear
hipX     = [0.4875 0.1625 -0.1625 -0.4875];
g.hipX   = hipX(g.pair);
g.hipY   = 0.2 * g.side;
g.coxa   = 0.5;
g.thigh  = 0.6;
g.tibia  = 0.7;
g.coxaElev = pi / 4;
g.qMin   = [-1.5708 -1.5708 -2.0];           % [hip thigh knee]
g.qMax   = [ 1.5708  1.5708  1.0];
g.maxVel = 2.0;
g.maxTorque = 50;
g.mass   = 15 + 8 * (1.2 + 1.4 + 0.9);       % body + 8 legs = 43 kg
g.radius = [0.03 0.0283 0.02];               % coxa, thigh, tibia collision radius
% Alternating support groups (tetrapod): A = L1 L3 R2 R4, B = L2 L4 R1 R3.
g.groupB = [0 1 0 1 0 1 0 1];
% Legs next to each other on the same side (the ones that can collide).
g.neighbours = [1 2; 2 3; 3 4; 5 6; 6 7; 7 8];
end
