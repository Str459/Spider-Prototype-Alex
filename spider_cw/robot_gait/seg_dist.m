function d = seg_dist(P1, Q1, P2, Q2)
%SEG_DIST Shortest distance between line segments P1-Q1 and P2-Q2 (row-wise,
%   N x 3 each). Standard closest-point method (Ericson, Real-Time Collision
%   Detection, 5.1.9), vectorised. Used to check leg-leg clearance.
d1 = Q1 - P1;
d2 = Q2 - P2;
r  = P1 - P2;
a = sum(d1 .^ 2, 2);
e = sum(d2 .^ 2, 2);
f = sum(d2 .* r, 2);
c = sum(d1 .* r, 2);
b = sum(d1 .* d2, 2);
den = a .* e - b .^ 2;
s = zeros(size(a));
ok = den > 1e-12;
s(ok) = clamp((b(ok) .* f(ok) - c(ok) .* e(ok)) ./ den(ok));
t = (b .* s + f) ./ e;
lo = t < 0;
t(lo) = 0;
s(lo) = clamp(-c(lo) ./ a(lo));
hi = t > 1;
t(hi) = 1;
s(hi) = clamp((b(hi) - c(hi)) ./ a(hi));
d = sqrt(sum((P1 + d1 .* s - P2 - d2 .* t) .^ 2, 2));
end

function x = clamp(x)
x = min(max(x, 0), 1);
end
