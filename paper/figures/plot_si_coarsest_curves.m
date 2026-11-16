function plot_si_coarsest_curves()
%PLOT_SI_COARSEST_CURVES Relative AD aliasing error against the grid ratio s.
% For the tolerances 1e-4 (P = 4, 5, 6) and 1e-8 (P = 7, 8, 10): the kernel
% of review_si_coarsest.py for each order and tolerance and B_P, in the
% limit of large boxes, and the psi_0 kernel with its bandwidth tuned on
% each grid, in a box of side 10 r_c (review_si_coarsest_curves.csv). The dotted line is the tolerance, and
% the markers are the coarsest ratios s* of the kernel, of B_P, and of the
% psi_0 kernel in the box (review_si_coarsest_record.json). Errors below
% 1e-11, near the floor of their computation, are not shown. Styles of
% paper_figure_style.m.
here=fileparts(mfilename('fullpath')); out=fullfile(here,'..','..','design','records');
S=paper_figure_style();
R=readtable(fullfile(out,'review_si_coarsest_curves.csv'));
data=jsondecode(fileread(fullfile(out,'review_si_coarsest_record.json')));
if iscell(data.results),results=data.results;else,results=num2cell(data.results);end
methods={'spline','prolate','design'};keys={'spline_sigma','prolate_sigma','sigma'};
names={'$B_P$','$\psi_0$ kernel','$\phi_P^*$'};
panels=[1e-4 4;1e-4 5;1e-4 6;1e-8 7;1e-8 8;1e-8 10];
floor_value=1e-11;
% About 450 pt wide, so that 10-point text prints at about 8 points in the 5.125-inch column.
f=figure('Visible','off','Color','w','Position',[80 80 600 450]);
layout=tiledlayout(f,2,3,'TileSpacing','compact','Padding','compact');
handles=gobjects(1,3);
for i=1:size(panels,1)
    tol=panels(i,1);P=panels(i,2);
    ax=nexttile(layout);
    T=R(abs(R.tolerance-tol)<1e-15 & R.P==P,:);[~,ii]=sort(T.s);T=T(ii,:);
    item=results{cellfun(@(x) abs(x.tolerance-tol)<1e-15 && x.P==P,results)};
    low=inf;high=0;
    for j=1:numel(methods)
        st=S.(methods{j});v=T.(methods{j});v(v<floor_value)=NaN;
        handles(j)=loglog(T.s,v,st.style,'Color',st.color,'LineWidth',S.width);hold on;
        low=min(low,min(v));high=max(high,max(v));
        star=item.(keys{j});
        if ~isempty(star) && isnumeric(star)
            loglog(star,tol,'o','Color',st.color,'MarkerFaceColor',st.color,'MarkerSize',4,'HandleVisibility','off');
        end
    end
    yline(tol,':','Color',S.gray,'LineWidth',1,'HandleVisibility','off');
    low=min(low,tol);high=max(high,tol);
    e0=floor(log10(low));e1=ceil(log10(high));
    grid on;xlim([1,4]);ylim(10.^[e0,e1]);
    % Every decade is labeled, or every other one on panels spanning more than six decades; the labels are
    % written out, since the automatic ones vanish when the ticks skip decades.
    step=1+(e1-e0>6);
    labels=arrayfun(@(e) sprintf('$10^{%d}$',e),e0:step:e1,'UniformOutput',false);
    set(ax,'FontSize',10,'XTick',[1 1.5 2 3 4],'XTickLabel',{'1','1.5','2','3','4'}, ...
        'YTick',10.^(e0:step:e1),'YTickLabel',labels);
    title(sprintf('$\\varepsilon = 10^{%d}$, $P = %d$',round(log10(tol)),P),'FontWeight','normal','FontSize',10);
    if i>3,xlabel('$s$','FontSize',11);end
    if mod(i,3)==1,ylabel('Relative AD aliasing error','FontSize',10);end
end
key=legend(handles,names,'NumColumns',3,'FontSize',9);key.Layout.Tile='south';
light_style(f);
exportgraphics(f,fullfile(here,'review_si_coarsest_curves.pdf'),'ContentType','vector');close(f);
end

function light_style(f)
set(findall(f,'Type','axes'),'Color','w','XColor','k','YColor','k','GridColor',[.6 .6 .6], ...
    'MinorGridColor',[.85 .85 .85],'GridAlpha',.4,'MinorGridAlpha',.3);
set(findall(f,'Type','legend'),'Box','on','Color','w','TextColor','k','EdgeColor',[.65 .65 .65]);
for ax=findall(f,'Type','axes')'
    set([ax.Title ax.XLabel ax.YLabel],'Color','k');
end
end
