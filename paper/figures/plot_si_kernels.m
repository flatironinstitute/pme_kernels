function plot_si_kernels()
%PLOT_SI_KERNELS Shapes phi(t) and transforms |phi_hat(q)| of the SI kernels, q = h xi.
% B_P, the psi_0 kernel tuned on the grid, and the optimized kernel of
% review_si_coarsest.py, for P = 5 at the tolerance 1e-3 and P = 4 at 1e-4,
% each on the coarsest grid s of the optimized kernel (review_si_figures.py:
% review_si_kernels_space.csv, review_si_kernels_frequency.csv,
% review_si_figures.json). Each kernel has unit integral. The dot marks the
% value from which the psi_0 kernel jumps to zero at t = P/2. In the
% transforms, the retained band |q| <= pi/s is shaded dark and its aliases
% around 2 pi m light; the dotted lines mark the Nyquist frequency q = pi and
% the multiples of 2 pi, where the circles mark the values of the psi_0
% kernel and B_P and the optimized kernel vanish. Styles of
% paper_figure_style.m.
here=fileparts(mfilename('fullpath')); out=fullfile(here,'..','..','design','records');
S=paper_figure_style();
X=readtable(fullfile(out,'review_si_kernels_space.csv'));
F=readtable(fullfile(out,'review_si_kernels_frequency.csv'));
info=jsondecode(fileread(fullfile(out,'review_si_figures.json')));
names={'spline','prolate','design'};
cases={'P5','P4'};
% About 450 pt wide, so that 10-point text prints at about 8 points in the 5.125-inch column.
f=figure('Visible','off','Color','w','Position',[80 80 600 440]);
layout=tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');
handles=gobjects(1,3);
for i=1:2
    key=cases{i};item=info.shapes.(key);P=item.P;s=item.sigma;
    heading=sprintf('$P = %d$, $\\varepsilon = 10^{%d}$',P,round(log10(item.tolerance)));
    % Physical space: the psi_0 kernel jumps at t = P/2, where B_P and the optimized kernel vanish.
    ax=nexttile(layout);
    t=X.([key '_t']);
    for j=[3 2 1]
        st=S.(names{j});v=X.([key '_' names{j}]);keep=v>0;
        handles(j)=semilogy(t(keep),v(keep),st.style,'Color',st.color,'LineWidth',S.width);hold on;
    end
    edge=item.prolate_edge;
    semilogy([P/2 P/2],[edge 1e-12],'-','Color',S.prolate.color,'LineWidth',S.width,'HandleVisibility','off');
    semilogy(P/2,edge,'o','Color',S.prolate.color,'MarkerFaceColor',S.prolate.color,'MarkerSize',4, ...
        'HandleVisibility','off');
    grid on;xlim([0,P/2]);ylim([1e-8,1]);
    set(ax,'FontSize',10,'XTick',0:.5:P/2,'YTick',10.^(-8:2:0));
    ylabel('$\phi(t)$','FontSize',11);
    title(heading,'FontWeight','normal','FontSize',10);
    if i==2,xlabel('$t$','FontSize',11);end
    % Fourier space.
    ax=nexttile(layout);
    q=F.q;band=pi/s;bottom=1e-12;top=3;
    shade(0,band,[.80 .80 .80],bottom,top);
    for m=1:2
        shade(2*pi*m-band,2*pi*m+band,[.92 .92 .92],bottom,top);
        xline(2*pi*m,':','Color',S.gray,'LineWidth',.8,'HandleVisibility','off');
    end
    xline(pi,':','Color',S.gray,'LineWidth',1.2,'HandleVisibility','off');
    for j=[3 2 1]
        st=S.(names{j});v=F.([key '_' names{j}]);v(v<bottom)=bottom;
        semilogy(q,v,st.style,'Color',st.color,'LineWidth',1.2);hold on;
    end
    v=F.([key '_prolate']);
    for m=1:2
        [~,k]=min(abs(q-2*pi*m));
        semilogy(q(k),v(k),'o','Color',S.prolate.color,'MarkerSize',5,'LineWidth',1.1,'HandleVisibility','off');
    end
    set(ax,'YScale','log','Layer','top');grid on;xlim([0,4*pi]);ylim([bottom,top]);
    set(ax,'FontSize',10,'XTick',0:pi:4*pi,'XTickLabel',{'$0$','$\pi$','$2\pi$','$3\pi$','$4\pi$'}, ...
        'YTick',10.^(-12:3:0));
    text(band/2,3e-12,'band','HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',8, ...
        'Color',[.2 .2 .2]);
    text(2*pi,3e-12,'aliases','HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',8, ...
        'Color',[.2 .2 .2],'BackgroundColor',[.92 .92 .92],'Margin',.5);
    text(pi*0.97,3e-12,'Nyquist','Rotation',90,'HorizontalAlignment','left','VerticalAlignment','bottom', ...
        'FontSize',8,'Color',[.2 .2 .2]);
    ylabel('$|\hat\phi(q)|$','FontSize',11);
    title(heading,'FontWeight','normal','FontSize',10);
    if i==2,xlabel('$q = h\xi$','FontSize',11);end
end
key=legend(handles,{'$B_P$','$\psi_0$ kernel','$\phi_P^*$'},'NumColumns',3,'FontSize',9);
key.Layout.Tile='south';
light_style(f);
exportgraphics(f,fullfile(here,'review_si_kernels.pdf'),'ContentType','vector');close(f);
end

function shade(a,b,color,bottom,top)
patch([a b b a],[bottom bottom top top],color,'EdgeColor','none','HandleVisibility','off');hold on;
end

function light_style(f)
set(findall(f,'Type','axes'),'Color','w','XColor','k','YColor','k','GridColor',[.6 .6 .6], ...
    'MinorGridColor',[.85 .85 .85],'GridAlpha',.4,'MinorGridAlpha',.3);
set(findall(f,'Type','legend'),'Box','on','Color','w','TextColor','k','EdgeColor',[.65 .65 .65]);
for ax=findall(f,'Type','axes')'
    set([ax.Title ax.XLabel ax.YLabel],'Color','k');
end
end
