function S = paper_figure_style()
%PAPER_FIGURE_STYLE Colors, line styles, and fonts shared by the figures of the paper.
% B_P is dashed blue, the psi_0 kernel solid orange, and the optimized kernel
% solid green in every figure; reference lines are gray. All text uses the
% LaTeX interpreter, so that the fonts match those of the paper.
set(groot,'defaultTextInterpreter','latex','defaultAxesTickLabelInterpreter','latex', ...
    'defaultLegendInterpreter','latex','defaultColorbarTickLabelInterpreter','latex');
S.spline=struct('color',[.12 .29 .55],'style','--');
S.prolate=struct('color',[.78 .27 .10],'style','-');
S.design=struct('color',[.06 .49 .35],'style','-');
S.gray=[.35 .35 .35];
S.width=1.4;
end
