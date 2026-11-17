function [score, info] = gait_fitness(c, s)
% Fitness of one gait chromosome for the Webots spider. Higher is better.
% s holds the settings (see run_ga.m). info has each part of the score.
%
% The only thing that is rewarded is walking speed. Everything else is a rule
% about what the real robot can do, and only costs points when it is broken.
% (When stability was also rewarded, the GA preferred standing still.)

[q, down, feet, outOfReach] = decode_gait(c, s.nFrames);

% Speed: a planted foot moves back by `stride` while it is down (duty x period
% seconds), and the body moves forward by the same amount.
info.speed = c(1) / (c(3) * c(4));            % m/s

% Every foot target must be reachable.
info.outOfReach = mean(outOfReach(:));

% Joints must stay inside the motor limits (with a small safety margin).
low = repmat(s.jointMin, 1, 8) + s.margin;
high = repmat(s.jointMax, 1, 8) - s.margin;
info.pastLimits = mean(max(0, q - high) + max(0, low - q), 'all');

% Joints must not turn faster than the motors can.
jointSpeed = abs(circshift(q, -1) - q) * s.nFrames / c(4);    % rad/s
info.tooFast = max(0, max(jointSpeed(:)) - s.maxJointSpeed);

% Self-collision: neighbouring legs on the same side must stay apart. We check
% the distance between their knees, the middle of their tibias and their feet.
P = leg_fk(q);
pts = cat(3, P(:, :, 3, :), (P(:, :, 3, :) + P(:, :, 4, :)) / 2, P(:, :, 4, :));   % n x 8 x 3 x 3
info.legsTooClose = 0;
neighbours = [1 2; 2 3; 3 4; 5 6; 6 7; 7 8];
for k = 1:size(neighbours, 1)
    for i = 1:3
        for j = 1:3
            d = vecnorm(pts(:, neighbours(k, 1), i, :) - pts(:, neighbours(k, 2), j, :), 2, 4);
            info.legsTooClose = info.legsTooClose + mean(max(0, s.legGap - d));
        end
    end
end

% Stability: the body centre (0,0) must be inside the planted feet. Seen from the
% centre, the feet must not leave a gap wider than s.maxGap (180 degrees or more
% would mean the robot tips over). Also at least 4 feet down.
bad = 0;
for f = 1:s.nFrames
    on = down(f, :);
    a = sort(atan2(feet(f, on, 2), feet(f, on, 1)));
    if sum(on) < 4 || max(diff([a, a(1) + 2 * pi])) > s.maxGap
        bad = bad + 1;
    end
end
info.unstable = bad / s.nFrames;

w = s.weights;
if isnan(s.targetSpeed)
    goal = w.speed * info.speed;                 % as fast as possible
else
    % walk at the target speed, changing the reference gait as little as possible
    info.change = mean(abs(c - s.reference) ./ s.range);
    goal = -w.target * abs(info.speed - s.targetSpeed) - w.change * info.change;
end
score = goal ...
      - w.reach * info.outOfReach ...
      - w.limits * info.pastLimits ...
      - w.fast * info.tooFast ...
      - w.legs * info.legsTooClose ...
      - w.unstable * info.unstable;
end
