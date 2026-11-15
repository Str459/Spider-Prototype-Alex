function fig = new_figure(width, height)
%NEW_FIGURE Off-screen white figure for saving plots.
%   Newer MATLAB releases default to a dark theme, which makes exported
%   PNGs hard to read; force the light theme where it exists.
fig = figure('Visible', 'off', 'Position', [100 100 width height], 'Color', 'w');
try
    theme(fig, 'light');
catch
    % Older MATLAB: no themes, already light.
end
end
