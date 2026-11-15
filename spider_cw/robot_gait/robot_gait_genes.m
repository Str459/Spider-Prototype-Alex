function genes = robot_gait_genes()
%ROBOT_GAIT_GENES The 11 gait parameters the GA evolves for the Webots robot.
%
%   The gait STRUCTURE is fixed so that it is a proper walk:
%   * two alternating support groups, A = L1 L3 R2 R4 and B = L2 L4 R1 R3
%     (the starting point), left and right mirrored;
%   * a planted foot moves in a straight line backwards relative to the body,
%     all planted feet at the same speed, so the body is pushed forwards
%     and no foot has to slide (the old gait's feet moved on arcs);
%   * a swinging foot lifts, moves forwards, and is placed down smoothly
%     (zero vertical speed at lift-off and touch-down);
%   * every joint angle comes from exact inverse kinematics, so all three
%     joints of a leg move together to follow the foot path.
%   The NUMBERS are evolved by the GA and judged by robot_gait_fitness.m.
%
%   gene          range           meaning
%   stride        0.05 - 0.45 m   distance a planted foot travels per step
%   stepHeight    0.06 - 0.20 m   foot lift at the top of a swing
%   duty          0.52 - 0.75     fraction of the cycle a foot is planted;
%                                 > 0.5 means both groups are down together
%                                 for a short time at every changeover
%   period        1.5  - 6.0 s    time for one full cycle
%   bodyHeight    0.42 - 0.60 m   body centre above the ground (body box is
%                                 0.3 m tall, so its belly clears 0.27-0.45 m).
%                                 The GA always takes the lowest allowed
%                                 height, so this bound decides it: at 0.30
%                                 the body hugged the floor (looks like
%                                 crawling); 0.42 walks as well as 0.38 in
%                                 Webots and matches the old gait's height.
%   reach         0.65 - 1.10 m   sideways distance of each foot from the
%                                 body centre line
%   spread        1.0  - 2.2      feet are placed at spread x their hip's
%                                 forward position (fans the legs out so
%                                 neighbours do not collide)
%   phaseB        0.35 - 0.65     delay of group B behind group A (fraction
%                                 of a cycle; 0.5 = pure alternation)
%   wave          -0.10 - 0.10    extra delay per leg pair front-to-back
%   xShift        -0.15 - 0.15 m  moves all footholds forwards/backwards
%                                 (balances the stance under the body)
%   speedMatch    0.0  - 1.0      ground-speed matching: at lift-off and
%                                 touch-down the swinging foot moves
%                                 backwards at this fraction of the stance
%                                 speed. 1 = the foot meets the ground at
%                                 zero ground speed (no scuff); 0 = it lands
%                                 still relative to the body, i.e. moving
%                                 forwards over the ground, and skids.
%                                 (Replaced "swingPad": in Webots the planted
%                                 feet still slid 0.2-0.4 m per metre walked.)

genes.names = {'stride', 'stepHeight', 'duty', 'period', 'bodyHeight', 'reach', ...
               'spread', 'phaseB', 'wave', 'xShift', 'speedMatch'};
genes.lb = [0.05 0.06 0.52 1.5 0.42 0.65 1.0 0.35 -0.10 -0.15 0.0];
genes.ub = [0.45 0.20 0.75 6.0 0.60 1.10 2.2 0.65  0.10  0.15 1.0];
genes.circular = false(1, numel(genes.lb));
genes.n = numel(genes.lb);
end
