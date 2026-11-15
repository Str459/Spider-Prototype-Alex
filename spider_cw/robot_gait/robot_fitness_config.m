function f = robot_fitness_config()
%ROBOT_FITNESS_CONFIG Settings and weights for robot_gait_fitness.m.
f.nFrames     = 100;     % frames per cycle while evolving (200 are exported)
f.limitMargin = 0.05;    % keep joints this far (rad) inside their limits
f.maxVel      = 1.6;     % rad/s, 80% of the motors' 2 rad/s
f.maxTorque   = 42;      % Nm, 84% of the motors' 50 Nm
f.minClear    = 0.04;    % m between neighbouring legs' surfaces
f.minMargin   = 0.15;    % m, body centre inside the support polygon
f.minSpeed    = 0.05;    % m/s; slower than this is "not walking"

f.w.speed     = 10;      % per m/s: the only thing that earns points
f.w.reach     = 50;      % per m out of reach
f.w.limit     = 20;      % per rad past a limit
f.w.vel       = 5;       % per rad/s above maxVel
f.w.torque    = 0.1;     % per Nm above maxTorque
f.w.collision = 50;      % per m of missing clearance
f.w.support   = 20;      % per fraction of frames with < 4 feet down
f.w.stability = 10;      % per m of missing margin
f.w.acc       = 0.002;   % per rad/s^2 RMS joint acceleration
f.w.still     = 5;       % flat penalty for not walking
f.w.contactSlip = 5;     % per m/s of foot ground speed at touch-down/lift-off
end
