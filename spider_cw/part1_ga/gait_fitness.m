function [score, comp, detail] = gait_fitness(chrom, fcfg, geom)
%GAIT_FITNESS Score a population of gait chromosomes (higher is better).
%   [score, comp] = GAIT_FITNESS(chrom, fcfg, geom)
%     chrom  P x 27 chromosomes (gait_genes.m)
%     fcfg   settings and weights (fitness_config.m)
%   returns
%     score  P x 1 fitness
%     comp   struct of P x 1 vectors, one per fitness term, so plots can
%            show *why* a gait scores what it does
%   [..., detail] also returns the per-frame data (feet, stance, stability
%   margin, body velocity) used by the plots and the animation.
%
%   There is no physics engine, so walking is judged geometrically:
%   * PROGRESS. Planted feet do not slide on the ground, so when they move
%     backwards relative to the body, the body has moved forwards by the
%     same amount. Body velocity in each frame = minus the mean movement of
%     the feet that are planted in that frame and the next one. Summed over
%     one cycle this gives the distance walked per cycle.
%   * REQUIREMENTS (only cost something when broken):
%       planted feet at ground height, no foot below the ground, swinging
%       feet lifted at least `clearance`, at least `minStance` feet down,
%       body centre at least `minMargin` inside the support polygon,
%       neighbouring feet at least `footGap` apart, little sideways drift,
%       planted feet moving together (low slip).
%   * Stability earns a small reward, capped at `marginCap`. Without the
%     cap a GA happily trades all forward motion for a slightly wider,
%     motionless stance ("standing still" is a strong local optimum).
%
%   Symmetry is not scored because the encoding is symmetric by design.
%   See docs/part1_fitness.txt for the full explanation and tuning notes.

P = size(chrom, 1);
F = fcfg.nFrames;
[angles, stance] = gait_decode(chrom, F, geom);

feet = spider_fk(reshape(angles, P * F, 24), geom);    % (P*F) x 8 x 3
X = reshape(feet(:, :, 1), P, F, 8);
Y = reshape(feet(:, :, 2), P, F, 8);
Z = reshape(feet(:, :, 3), P, F, 8);
dz = Z - fcfg.groundZ;                                  % height above ground
swing = ~stance;

% --- Ground contact ------------------------------------------------------
nPlanted  = max(sum(stance, [2 3]), 1);
groundErr = sum(abs(dz) .* stance, [2 3]) ./ nPlanted;  % mean |height| of planted feet
penetr    = mean(max(0, -dz), [2 3]);                   % mean depth below ground

dzSwing = dz;
dzSwing(stance) = -Inf;
topOfSwing = max(dzSwing, [], 2);                       % P x 1 x 8
topOfSwing(isinf(topOfSwing)) = 0;
clearDef = mean(max(0, fcfg.clearance - topOfSwing), 3);

% --- Static stability ----------------------------------------------------
[margin, nStance] = support_margin(reshape(X, P * F, 8), ...
                                   reshape(Y, P * F, 8), ...
                                   reshape(stance, P * F, 8));
margin  = reshape(margin, P, F);
nStance = reshape(nStance, P, F);
minMarg   = min(margin, [], 2);
instab    = mean(max(0, fcfg.minMargin - margin), 2);
stanceDef = mean(nStance < fcfg.minStance, 2);          % fraction of frames short of feet

% --- Progress from planted feet -----------------------------------------
both = stance & circshift(stance, -1, 2);               % planted now and next frame
dX = circshift(X, -1, 2) - X;                           % frame F wraps to frame 1
dY = circshift(Y, -1, 2) - Y;
cnt = max(sum(both, 3), 1);
mdX = sum(dX .* both, 3) ./ cnt;                        % P x F mean foot motion
mdY = sum(dY .* both, 3) ./ cnt;
vx = -mdX;                                              % body velocity per frame
vy = -mdY;
progress = sum(vx, 2);                                  % distance per cycle
lateral  = abs(sum(vy, 2));
slipMag  = hypot(dX - mdX, dY - mdY) .* both;           % each foot vs the average
slip     = sum(sum(slipMag, 3) ./ cnt, 2);

% --- Neighbouring feet must not collide ---------------------------------
collision = zeros(P, 1);
for k = 1:size(geom.neighbours, 1)
    a = geom.neighbours(k, 1);
    b = geom.neighbours(k, 2);
    gap = hypot(X(:, :, a) - X(:, :, b), Y(:, :, a) - Y(:, :, b));
    collision = collision + mean(max(0, fcfg.footGap - gap), 2);
end

% --- Effort: mean squared joint speed (radians per cycle) ---------------
speed  = (circshift(angles, -1, 2) - angles) * F;
effort = mean(speed .^ 2, [2 3]);

% --- Combine -------------------------------------------------------------
w = fcfg.w;
if isnan(fcfg.targetSpeed)
    goal = w.progress * progress;
else
    goal = -w.target * abs(progress - fcfg.targetSpeed);
end
score = goal ...
      + w.margin    * min(minMarg, fcfg.marginCap) ...
      - w.ground    * groundErr ...
      - w.pen       * penetr ...
      - w.clear     * clearDef ...
      - w.instab    * instab ...
      - w.stance    * stanceDef ...
      - w.slip      * slip ...
      - w.lateral   * lateral ...
      - w.collision * collision ...
      - w.effort    * effort;

comp = struct('progress', progress, 'lateral', lateral, 'slip', slip, ...
    'groundErr', groundErr, 'penetration', penetr, 'clearanceDeficit', clearDef, ...
    'minMargin', minMarg, 'instability', instab, 'stanceDeficit', stanceDef, ...
    'collision', collision, 'effort', effort, 'score', score);

if nargout > 2
    detail = struct('angles', angles, 'stance', stance, ...
        'feet', cat(4, X, Y, Z), 'margin', margin, 'nStance', nStance, ...
        'vx', vx, 'vy', vy, 'groundZ', fcfg.groundZ);
end
end
