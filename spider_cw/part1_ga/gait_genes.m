function genes = gait_genes()
%GAIT_GENES Definition of the 27-gene gait chromosome (Stage B).
%   genes.names, genes.lb, genes.ub  1 x 27 gene names and bounds
%   genes.circular                   1 x 27 logical, true for phase genes
%                                    (they wrap round: 0.99 is next to 0.01)
%
%   The chromosome is a real-valued vector. It does not store 300 poses;
%   gait_decode.m expands it into the 300 x 24 joint-angle gait.
%
%   Genes 1-24: six per front-to-back leg pair (front, midfront, midrear,
%   rear). The left and right leg of a pair share them, mirrored, so the
%   gait is symmetric by construction.
%       coxaOffset   mean "forward" coxa angle  (+ = foot further forward)
%       coxaSwing    coxa amplitude = half the stride, in radians
%       femurStance  femur angle while the foot is on the ground
%       femurLift    extra femur angle at the top of a swing (lifts foot)
%       tibiaStance  tibia angle while the foot is on the ground
%       tibiaLift    extra tibia angle at the top of a swing
%   Genes 25-27: timing shared by all legs.
%       phaseLR      delay of each right leg behind its left partner
%                    (fraction of a cycle)
%       phaseFB      delay of each pair behind the pair in front of it
%       duty         fraction of the cycle each foot spends on the ground
%
%   Bounds keep joints in a physically sensible range and stop the coxa
%   swinging so far that neighbouring legs pass through each other.
%   See docs/part1_encoding.txt for the reasoning.

pairNames = {'front', 'midfront', 'midrear', 'rear'};
perPair   = {'coxaOffset', 'coxaSwing', 'femurStance', 'femurLift', 'tibiaStance', 'tibiaLift'};
pairLb    = [-0.5, 0.0, -1.2, 0.0, -2.0, -0.8];
pairUb    = [ 0.5, 0.5,  0.6, 0.8,  0.0,  0.8];

names = {};
for k = 1:4
    for j = 1:numel(perPair)
        names{end + 1} = [pairNames{k} '_' perPair{j}]; %#ok<AGROW>
    end
end
names = [names, {'phaseLR', 'phaseFB', 'duty'}];

genes.names    = names;
genes.lb       = [repmat(pairLb, 1, 4), 0, 0, 0.5];
genes.ub       = [repmat(pairUb, 1, 4), 1, 1, 0.9];
genes.circular = [false(1, 24), true, true, false];
genes.n        = numel(genes.lb);
end
