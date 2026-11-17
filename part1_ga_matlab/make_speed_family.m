% Part 2 training data: the GA makes one gait for each target speed (m/s).
% Run run_ga first (the fastest gait is the starting point for every speed).
for v = [0.10 0.15 0.20 0.25 0.30 0.35]
    run_ga('tournament', 'twopoint', 1, v);
end
