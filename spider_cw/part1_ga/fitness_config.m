function f = fitness_config()
%FITNESS_CONFIG Tunable settings and weights of the gait fitness function.
%   Every value can be overridden from run_gait_ga, for example
%       run_gait_ga('groundZ', -0.5, 'targetSpeed', 1.0)
%   See docs/part1_fitness.txt for why each term exists.

f.nFrames     = 300;     % frames per cycle. Scoring the same 300 frames that
                         % are exported matters: at 60 frames, short moments
                         % with only 3 feet down fell between samples.
f.groundZ     = -0.4;    % height of the ground below the body centre
f.clearance   = 0.15;    % how high a swinging foot must lift
f.minMargin   = 0.30;    % required stability margin (body centre to polygon edge)
f.marginCap   = 0.30;    % stability above this earns nothing extra
f.minStance   = 4;       % feet that must be on the ground in every frame
f.footGap     = 0.35;    % closest two neighbouring feet may get
f.targetSpeed = NaN;     % NaN = walk as far as possible per cycle;
                         % a number = walk exactly this far (used to make
                         % training data at different speeds for Part 2)

% Weights. "Goal first": progress is the only term that keeps paying,
% everything else is a requirement that only costs when it is violated.
f.w.progress  = 3;       % distance walked per cycle
f.w.target    = 5;       % |distance - targetSpeed| when a target is set
f.w.margin    = 1;       % small reward for stability, capped at marginCap
f.w.ground    = 20;      % planted feet not on the ground
f.w.pen       = 30;      % any foot below the ground
f.w.clear     = 10;      % swinging foot not lifted high enough
f.w.instab    = 20;      % frames where the margin is below minMargin
f.w.stance    = 50;      % fraction of frames with fewer than minStance feet down
                         % (one bad frame in 300 costs 0.17: enough to matter)
f.w.slip      = 1;       % planted feet sliding against each other
f.w.lateral   = 2;       % sideways drift per cycle
f.w.collision = 20;      % neighbouring feet closer than footGap
f.w.effort    = 0.01;    % mean squared joint speed (energy proxy)
end
