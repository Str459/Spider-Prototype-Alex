function [T, stance, period] = robot_gait_decode(chrom, F, g)
%ROBOT_GAIT_DECODE Gait genes -> foot targets for every leg and frame.
%   [T, stance, period] = ROBOT_GAIT_DECODE(chrom, F, g)
%     chrom   P x 11 (robot_gait_genes.m), F frames per cycle
%   returns
%     T       P x F x 8 x 3 foot targets in the body frame
%     stance  P x F x 8 logical, true = foot planted
%     period  P x 1 seconds per cycle

P = size(chrom, 1);
r = @(v) reshape(v, P, 1, 1);
S     = r(chrom(:, 1));
H     = r(chrom(:, 2));
beta  = r(chrom(:, 3));
period = chrom(:, 4);
h     = r(chrom(:, 5));
reach = r(chrom(:, 6));
spread = r(chrom(:, 7));
phB   = chrom(:, 8);
wave  = chrom(:, 9);
xs    = r(chrom(:, 10));
match = r(chrom(:, 11));

% Each leg's delay: group B starts phaseB later; each pair further back
% starts `wave` later (0 = all legs of a group exactly together).
delay = g.groupB .* phB + (g.pair - 1) .* wave;                 % P x 8
t = (0:F - 1) / F;
phase = mod(reshape(t, 1, F) - reshape(delay, P, 1, 8), 1);    % P x F x 8
stance = phase < beta;
swing = ~stance;

sp = phase ./ beta;                                  % 0 -> 1 through stance
u  = (phase - beta) ./ (1 - beta);                   % 0 -> 1 through swing

% Stance: straight back at constant speed (all planted feet identical, so
% none has to slide). Swing: cubic Hermite from -S/2 to +S/2 whose speed at
% both ends is speedMatch x the stance speed (backwards), so the foot
% leaves and meets the ground with little or no speed over the ground.
m  = -match .* S .* (1 - beta) ./ beta;              % end slope in swing units
h00 = 2 * u .^ 3 - 3 * u .^ 2 + 1;
h10 = u .^ 3 - 2 * u .^ 2 + u;
h01 = -2 * u .^ 3 + 3 * u .^ 2;
h11 = u .^ 3 - u .^ 2;
xSwing = -S / 2 .* h00 + m .* h10 + S / 2 .* h01 + m .* h11;
xrel = stance .* (S / 2 - S .* sp) + swing .* xSwing;
% Swing height: (1 - cos 2*pi*u)/2 has zero vertical speed at both ends,
% so the foot leaves and meets the ground gently.
lift = swing .* H .* (1 - cos(2 * pi * u)) / 2;

x0 = spread .* reshape(g.hipX, 1, 1, 8) + xs;
X = x0 + xrel;
Y = repmat(reach .* reshape(g.side, 1, 1, 8), 1, F, 1);
Z = -h + lift;
T = cat(4, X, Y, Z);
end
