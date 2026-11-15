function [angles, stance, phase] = gait_decode(chrom, nFrames, geom)
%GAIT_DECODE Expand gait chromosomes into joint-angle trajectories.
%   [angles, stance, phase] = GAIT_DECODE(chrom, nFrames, geom)
%     chrom   P x 27 chromosomes (see gait_genes.m)
%     nFrames frames in one gait cycle (300 for the final gait)
%   returns
%     angles  P x nFrames x 24  joint angles, MATLAB leg order, radians
%     stance  P x nFrames x 8   true where the foot is meant to be planted
%     phase   P x nFrames x 8   each leg's position in its own cycle, [0,1)
%
%   Each leg runs the same step cycle, shifted in time by the timing genes:
%     leg delay = (pair - 1) * phaseFB  (+ phaseLR for right legs)
%   Within a cycle a leg spends a fraction "duty" in STANCE, then SWINGS.
%
%   Stance (foot on the ground, pushing the body forward):
%     the coxa sweeps the foot from +swing to -swing at constant speed.
%     Femur and tibia are held still. A coxa turn is a rotation about the
%     vertical axis, so it moves the foot horizontally but cannot change
%     its height: a planted foot stays on the ground for the whole stance.
%   Swing (foot in the air, returning):
%     the coxa brings the foot forward again along a smooth cosine, while
%     femur and tibia add a half-sine "lift" that is zero at lift-off and
%     touch-down. So the foot clears the ground in the middle of the swing.
%
%   Positions are continuous everywhere, so frame 300 joins back to frame 1
%   and the gait can loop forever.

P = size(chrom, 1);
F = nFrames;
pair = geom.pair;                         % 1 x 8

% Per-leg copies of the per-pair genes: each is P x 8.
col = @(j) chrom(:, (pair - 1) * 6 + j);
c0 = col(1);  A1 = col(2);  f0 = col(3);
A2 = col(4);  t0 = col(5);  A3 = col(6);

phaseLR = chrom(:, 25);
phaseFB = chrom(:, 26);
duty    = chrom(:, 27);

delay = (pair - 1) .* phaseFB + (~geom.isLeft) .* phaseLR;     % P x 8

t = (0:F - 1) / F;                                             % 1 x F
phase = mod(reshape(t, 1, F) - reshape(delay, P, 1, 8), 1);    % P x F x 8
beta  = reshape(duty, P, 1, 1);
stance = phase < beta;
swing  = ~stance;

s = phase ./ beta;                % 0 -> 1 through the stance part
u = (phase - beta) ./ (1 - beta); % 0 -> 1 through the swing part

A1r = reshape(A1, P, 1, 8);
fwd  = stance .* (A1r .* (1 - 2 * s)) + swing .* (-A1r .* cos(pi * u));
lift = swing .* sin(pi * u);

fwdSign = reshape(geom.fwdSign, 1, 1, 8);
theta1 = fwdSign .* (reshape(c0, P, 1, 8) + fwd);
theta2 = reshape(f0, P, 1, 8) + reshape(A2, P, 1, 8) .* lift;
theta3 = reshape(t0, P, 1, 8) + reshape(A3, P, 1, 8) .* lift;

angles = zeros(P, F, 24);
angles(:, :, 1:3:end) = theta1;
angles(:, :, 2:3:end) = theta2;
angles(:, :, 3:3:end) = theta3;
end
