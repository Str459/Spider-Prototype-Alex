function [angles, down, feet, outOfReach] = decode_gait(c, nFrames)
% Turns one chromosome (8 numbers) into a full gait for the Webots spider.
%   angles      nFrames x 24 motor angles, legs L1 L2 L3 L4 R4 R3 R2 R1 (hip, thigh, knee)
%   down        nFrames x 8, true when that foot is on the ground
%   feet        nFrames x 8 x 3, where each foot should be (body frame, metres)
%   outOfReach  nFrames x 8, how far a foot target is outside the leg's reach
%
% Genes:
%   1 stride      how far a planted foot moves back during one step (m)
%   2 stepHeight  how high a swinging foot is lifted (m)
%   3 duty        fraction of the cycle a foot is on the ground
%   4 period      seconds for one full cycle
%   5 bodyHeight  height of the body centre above the ground (m)
%   6 reach       sideways distance of the feet from the middle of the body (m)
%   7 spread      each foot is placed at spread x its hip's forward position (fans the legs out)
%   8 xShift      moves all footholds forwards or backwards (m)
%
% The legs move in two groups, A = L1 L3 R2 R4 and B = L2 L4 R1 R3, half a cycle
% apart. Each group is 4 feet that make a stable shape around the body. With
% duty > 0.5 both groups are down together for a moment at every change-over.
%
% We describe where the FEET go and then use inverse kinematics (leg_ik.m) to get
% the joint angles. On this robot the hip axis is tilted, so turning only the hip
% would also lift or lower the foot. Working from foot positions avoids that.

stride = c(1);
height = c(2);
duty = c(3);
bodyHeight = c(5);
reach = c(6);
spread = c(7);
xShift = c(8);

hipX = [0.4875 0.1625 -0.1625 -0.4875 -0.4875 -0.1625 0.1625 0.4875];
side = [1 1 1 1 -1 -1 -1 -1];
groupB = [0 1 0 1 0 1 0 1];

feet = zeros(nFrames, 8, 3);
down = false(nFrames, 8);
for leg = 1:8
    p = mod((0:nFrames - 1)' / nFrames - 0.5 * groupB(leg), 1);   % where this leg is in its step (0 to 1)
    onGround = p < duty;
    s = p / duty;                    % 0 to 1 while on the ground
    u = (p - duty) / (1 - duty);     % 0 to 1 while in the air

    % On the ground the foot moves straight back at a steady speed. All planted
    % feet move the same way, so the body is pushed forward and no foot has to slide.
    xGround = stride / 2 - stride * s;
    % In the air the foot comes forward again on a smooth (cubic) curve. At lift-off
    % and touch-down it moves back at the same speed as the ground, so it doesn't scuff.
    v = -stride * (1 - duty) / duty;
    xAir = -stride / 2 * (2 * u.^3 - 3 * u.^2 + 1) + v * (u.^3 - 2 * u.^2 + u) ...
         + stride / 2 * (-2 * u.^3 + 3 * u.^2) + v * (u.^3 - u.^2);
    lift = height * (1 - cos(2 * pi * u)) / 2;   % 0 at both ends, stepHeight in the middle

    feet(:, leg, 1) = spread * hipX(leg) + xShift + onGround .* xGround + ~onGround .* xAir;
    feet(:, leg, 2) = reach * side(leg);
    feet(:, leg, 3) = -bodyHeight + ~onGround .* lift;
    down(:, leg) = onGround;
end
[angles, outOfReach] = leg_ik(feet);
end
