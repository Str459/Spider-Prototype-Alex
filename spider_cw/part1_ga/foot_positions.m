function [x, y, z] = foot_positions(angles)
% Foot positions for every pose in angles (n x 24). x, y and z are n x 8.
% Same model as the lecturer's forward_leg_kinematics2.m, but written out
% directly so it can do all legs and frames at once (the GA needs millions
% of these, and calling the lecturer's function one leg at a time is far too slow).
%
% If you work through the lecturer's rotations, each leg stays in a vertical
% plane pointing at (base angle + theta1). The coxa is tilted up 30 degrees,
% the femur is at 30 deg + theta2 and the tibia at femur + theta3. The lecturer's
% code flips the direction of the tibia rotation once the femur points back past
% vertical, which is what the sign(cos) term copies.
% run_ga.m checks this against forward_leg_kinematics2.m.

L = [1.2 0.7 1.0];                                   % coxa, femur, tibia
base = deg2rad([45 75 105 135 -135 -105 -75 -45]);  % where the legs join the body
bx = 1.5 * cos(base);                                % body is an ellipse 1.5 x 1.0
by = 1.0 * sin(base);

heading = base + angles(:, 1:3:end);
femur = deg2rad(30) + angles(:, 2:3:end);
tibia = femur + sign(cos(femur)) .* angles(:, 3:3:end);

reach = L(1) * cosd(30) + L(2) * cos(femur) + L(3) * cos(tibia);   % horizontal distance from the leg base
x = bx + reach .* cos(heading);
y = by + reach .* sin(heading);
z = L(1) * sind(30) + L(2) * sin(femur) + L(3) * sin(tibia);
end
