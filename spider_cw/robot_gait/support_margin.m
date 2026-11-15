function [margin, nStance] = support_margin(x, y, planted)
%SUPPORT_MARGIN Static stability margin of many poses at once.
%   [margin, nStance] = SUPPORT_MARGIN(x, y, planted)
%     x, y     M x 8 foot positions seen from above (body frame)
%     planted  M x 8 logical, which feet are on the ground
%   returns
%     margin   M x 1 distance from the body centre to the nearest edge of
%              the support polygon. Positive = statically stable,
%              negative = the spider would tip over. -1 if < 3 feet down.
%     nStance  M x 1 number of planted feet
%
%   We assume the centre of mass is at the body centre (0,0); there is no
%   mass model in the provided code. Planted feet are sorted by angle around
%   the body centre and joined in that order. For each edge p -> q the
%   signed distance of the centre is cross(p, q) / |q - p|. If any gap
%   between neighbouring feet is wider than 180 degrees the centre lies
%   outside and the distance comes out negative. If a planted foot sits
%   inside the polygon (non-convex order) this gives a slightly smaller
%   value than the true convex hull, i.e. it errs on the safe side.

M = size(x, 1);
ang = atan2(y, x);
ang(~planted) = NaN;
[~, order] = sort(ang, 2);                    % NaNs (lifted feet) go last
rows = repmat((1:M)', 1, 8);
idx = sub2ind([M 8], rows, order);
xs = x(idx);
ys = y(idx);
nStance = sum(planted, 2);

d = inf(M, 8);
for k = 1:8
    nxt = (k + 1) * ones(M, 1);
    nxt(k >= nStance) = 1;                    % last planted foot joins the first
    qx = xs(sub2ind([M 8], (1:M)', nxt));
    qy = ys(sub2ind([M 8], (1:M)', nxt));
    px = xs(:, k);
    py = ys(:, k);
    len = hypot(qx - px, qy - py);
    dk = (px .* qy - py .* qx) ./ max(len, 1e-9);
    dk(k > nStance) = Inf;                    % not a real edge
    d(:, k) = dk;
end
margin = min(d, [], 2);
margin(nStance < 3) = -1;                     % a point or a line cannot hold the body up
end
