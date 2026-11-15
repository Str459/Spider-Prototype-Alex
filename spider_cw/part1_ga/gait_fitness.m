function [score, info] = gait_fitness(c, s)
% Fitness of one gait chromosome. Higher is better.
% s holds the settings (see run_ga.m). info has each part of the score.
%
% There is no physics engine, so walking is judged from the foot positions.
% The only thing that is rewarded is distance walked. Everything else is a
% rule that only costs points when it is broken, otherwise the GA learns to
% stand still in a very stable pose instead of walking.

[angles, down] = decode_gait(c, s.nFrames);
[x, y, z] = foot_positions(angles);
h = z - s.groundZ;                       % height of each foot above the ground

% Distance walked. Planted feet don't slide, so if they move back by d
% the body has moved forward by d.
both = down & circshift(down, -1);       % planted in this frame and the next
dx = circshift(x, -1) - x;
dy = circshift(y, -1) - y;
n = max(sum(both, 2), 1);
meanDx = sum(dx .* both, 2) ./ n;
meanDy = sum(dy .* both, 2) ./ n;
info.distance = -sum(meanDx);

% Slip: all planted feet should move together. Feet moving on different
% arcs have to slide over the ground.
info.slip = sum(sum(hypot(dx - meanDx, dy - meanDy) .* both, 2) ./ n);

% Planted feet on the ground, and no foot under it.
info.groundError = mean(abs(h(down)));
info.belowGround = mean(max(0, -h(:)));

% Swinging feet must lift at least s.clearance, but not more than s.maxLift
% (without this the GA kicks the back legs up very high, which looks silly).
hSwing = h;
hSwing(down) = -Inf;
top = max(hSwing);
top(isinf(top)) = 0;
info.lowSwing = mean(max(0, s.clearance - top));
info.highSwing = mean(max(0, top - s.maxLift));

% Stability: at least 4 feet down, and the body centre (0,0) inside the
% planted feet. Seen from the centre, the feet must not leave a gap wider
% than s.maxGap; a gap of 180 degrees or more means the body would tip over.
bad = 0;
for f = 1:s.nFrames
    on = down(f, :);
    if sum(on) < 4
        bad = bad + 1;
        continue
    end
    a = sort(atan2(y(f, on), x(f, on)));
    gaps = diff([a, a(1) + 2 * pi]);
    if max(gaps) > s.maxGap
        bad = bad + 1;
    end
end
info.unstable = bad / s.nFrames;         % fraction of frames that are unstable

% Neighbouring feet must not hit each other.
info.feetTooClose = 0;
for leg = 1:8
    other = mod(leg, 8) + 1;
    d = hypot(x(:, leg) - x(:, other), y(:, leg) - y(:, other));
    info.feetTooClose = info.feetTooClose + mean(max(0, s.footGap - d));
end

w = s.weights;
score = w.distance * info.distance ...
      - w.slip * info.slip ...
      - w.ground * info.groundError ...
      - w.below * info.belowGround ...
      - w.lift * info.lowSwing ...
      - w.lift * info.highSwing ...
      - w.unstable * info.unstable ...
      - w.feet * info.feetTooClose;
end
