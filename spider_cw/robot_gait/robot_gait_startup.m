function [T, stance] = robot_gait_startup(chrom, F, M, g)
%ROBOT_GAIT_STARTUP Foot targets for starting to walk from standing.
%   [T, stance] = ROBOT_GAIT_STARTUP(chrom, F, M, g) returns (M+1)*F frames
%   (F per cycle) for ONE chromosome: T is frames x 8 x 3, stance frames x 8.
%
%   Starting the loop directly from standing would mean every planted foot
%   jumping from the centre of its stride to where the loop expects it: all
%   eight feet dragged across the floor at once. Instead the stride grows a
%   step at a time: a leg's m-th stance uses stride k(m) = min(1, m/M) x the
%   evolved stride, and it only changes while that foot is in the air (the
%   swing goes from the end of one stance to the start of the next). Step 0
%   is marching on the spot, step M and later are the evolved gait. After
%   M+1 cycles every leg is at full stride and the last frame leads straight
%   into frame 1 of joint_table.csv.

S = chrom(1);  H = chrom(2);  beta = chrom(3);  h = chrom(5);
reach = chrom(6);  spread = chrom(7);  phB = chrom(8);  wave = chrom(9);
xs = chrom(10);  match = chrom(11);

delay = g.groupB * phB + (g.pair - 1) * wave;               % 1 x 8
tau = (0:(M + 1) * F - 1)' / F;                            % cycles since start
c = floor(tau - delay);                                     % cycle index per leg
phase = (tau - delay) - c;
m = c - floor(-delay);                                      % 0 = step in progress at start
k  = @(m) min(1, max(0, m) / M);
kA = k(m);                                                  % this stance / last stance
kB = k(m + 1);                                              % next stance

stance = phase < beta;
sp = phase / beta;
u  = (phase - beta) / (1 - beta);
slope = -match * S * (1 - beta) / beta;
h00 = 2 * u .^ 3 - 3 * u .^ 2 + 1;
h10 = u .^ 3 - 2 * u .^ 2 + u;
h01 = -2 * u .^ 3 + 3 * u .^ 2;
h11 = u .^ 3 - u .^ 2;
xSwing = -kA * S / 2 .* h00 + kA * slope .* h10 + kB * S / 2 .* h01 + kB * slope .* h11;
xrel = stance .* kA .* (S / 2 - S * sp) + ~stance .* xSwing;
lift = ~stance .* H .* (1 - cos(2 * pi * u)) / 2;

n = numel(tau);
X = spread * g.hipX + xs + xrel;
Y = repmat(reach * g.side, n, 1);
Z = -h + lift;
T = cat(3, X, Y, Z);
end
