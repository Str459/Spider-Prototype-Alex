function [angles, down] = decode_gait(c, nFrames)
% Turns one chromosome (27 numbers) into a full gait.
%   angles  nFrames x 24 joint angles in radians, leg order L1 L2 L3 L4 R4 R3 R2 R1
%   down    nFrames x 8, true when that foot should be on the ground
%
% Chromosome layout:
%   genes 1-24  six genes for each pair of legs (front, 2nd, 3rd, back).
%               The left and right leg of a pair share them (mirrored),
%               so the gait is always symmetric.
%                 1 coxa offset   where the leg points on average
%                 2 coxa swing    how far the leg sweeps (half a step)
%                 3 femur stance  femur angle while the foot is down
%                 4 femur lift    extra femur angle at the top of a swing
%                 5 tibia stance  tibia angle while the foot is down
%                 6 tibia lift    extra tibia angle at the top of a swing
%   gene 25     phaseLR: how far (fraction of a cycle) a right leg is behind its left partner
%   gene 26     phaseFB: how far each pair is behind the pair in front of it
%   gene 27     duty: fraction of the cycle a foot spends on the ground

pairOf  = [1 2 3 4 4 3 2 1];
isRight = [0 0 0 0 1 1 1 1];
fwdSign = [-1 -1 -1 -1 1 1 1 1];   % a positive theta1 moves a left foot back but a right foot forward
duty = c(27);

angles = zeros(nFrames, 24);
down = false(nFrames, 8);
for leg = 1:8
    g = c((pairOf(leg) - 1) * 6 + (1:6));
    delay = (pairOf(leg) - 1) * c(26) + isRight(leg) * c(25);
    p = mod((0:nFrames - 1)' / nFrames - delay, 1);   % where this leg is in its own step (0 to 1)
    onGround = p < duty;

    % On the ground the coxa sweeps the foot from front to back at a steady speed.
    % Femur and tibia stay still, and turning the coxa doesn't change the foot's
    % height, so the foot stays on the ground the whole time.
    % In the air the foot is brought forward again and lifted (sin is 0 at both ends).
    s = p / duty;
    u = (p - duty) / (1 - duty);
    sweep = onGround .* g(2) .* (1 - 2 * s) + ~onGround .* (-g(2) * cos(pi * u));
    lift = ~onGround .* sin(pi * u);

    angles(:, 3 * leg - 2) = fwdSign(leg) * (g(1) + sweep);
    angles(:, 3 * leg - 1) = g(3) + g(4) * lift;
    angles(:, 3 * leg)     = g(5) + g(6) * lift;
    down(:, leg) = onGround;
end
end
