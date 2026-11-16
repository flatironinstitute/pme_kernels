function plot_assigned_charge()
%PLOT_ASSIGNED_CHARGE Charge that a unit charge assigns to the grid, P = 4.
% The deviation S(y_alpha/h)-1 of S(y_alpha/h) = sum_l phi(l-y_alpha/h) and its
% derivative over one grid cell for the cubic B-spline, the tuned psi_0
% kernel, and the kernel of si_coarsest.py for the tolerance 1e-4,
% as computed by si_coarsest.py --charge (assigned_charge.csv).
% Styles of paper_figure_style.m.
here=fileparts(mfilename('fullpath')); out=fullfile(here,'..','..','design','records');
S=paper_figure_style();
T=readtable(fullfile(out,'assigned_charge.csv'));
info=jsondecode(fileread(fullfile(out,'assigned_charge.json')));
names={'spline','prolate','design'};
% The bandwidth of the psi_0 kernel, info.prolate_cw, is given in the caption.
labels={'$B_4$','$\psi_0$ kernel','$\phi_4^*$'};
theta=T.theta;n=numel(theta);
% The derivative for the psi_0 kernel jumps at y_alpha/h = 0, where the ends of its support cross grid points.
gap=[theta(1:n/2);NaN;theta(n/2+1:end)];
% About 450 pt wide, so that 10-point text prints at about 8 points in the 5.125-inch column.
f=figure('Visible','off','Color','w','Position',[80 80 600 290]);
layout=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
handles=gobjects(1,3);
for panel=1:2
    nexttile(layout);
    suffix={'_S','_dS'};scale=[1e4 1e3];
    % The optimized kernel first, so that the dashed B_4 stays visible on the same zero line, and with circles,
    % so that both curves can be seen where they coincide.
    for k=[3 2 1]
        y=T.([names{k} suffix{panel}])*scale(panel);st=S.(names{k});
        if k==2 && panel==2
            handles(k)=plot(gap,[y(1:n/2);NaN;y(n/2+1:end)],st.style,'Color',st.color,'LineWidth',S.width);
        elseif k==3
            handles(k)=plot(theta,y,st.style,'Color',st.color,'LineWidth',S.width,'Marker','o', ...
                'MarkerIndices',n/20:n/10:n,'MarkerSize',5,'MarkerFaceColor','w');
        else
            handles(k)=plot(theta,y,st.style,'Color',st.color,'LineWidth',S.width);
        end
        hold on;
    end
    grid on;xlim([-.5,.5]);set(gca,'FontSize',10);xlabel('$y_\alpha/h$','FontSize',11);
    if panel==1
        ylabel('$10^4\,(S(y_\alpha/h)-1)$','FontSize',11);title('Assigned charge, $P=4$','FontWeight','normal');
    else
        ylabel('$10^3\,S''(y_\alpha/h)$','FontSize',11);title('Derivative','FontWeight','normal');
    end
end
key=legend(handles,labels,'NumColumns',3,'FontSize',9);
key.Layout.Tile='south';
light_style(f);
exportgraphics(f,fullfile(here,'assigned_charge.pdf'),'ContentType','vector');close(f);
end

function light_style(f)
set(findall(f,'Type','axes'),'Color','w','XColor','k','YColor','k','GridColor',[.6 .6 .6], ...
    'MinorGridColor',[.85 .85 .85],'GridAlpha',.4,'MinorGridAlpha',.3);
set(findall(f,'Type','legend'),'Box','on','Color','w','TextColor','k','EdgeColor',[.65 .65 .65]);
for ax=findall(f,'Type','axes')'
    set([ax.Title ax.XLabel ax.YLabel],'Color','k');
end
end
