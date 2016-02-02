function plotFig(varargin)

color_arr = {'k' 'b' 'r' 'm' 'g' 'r' 'w' 'y' 'b'};
%color_arr = { 'r'  'k' 'r' 'r' 'k'  'k'  'c'   'b'};
%dot_arr = { 'x' 's'  's'  's'  'x' 's' 'x'    '^'};
%dot_arr = {'' '' '' '' '' '' '' ''    ''};
line_arr = {'-' '-' '-' '-' '-' '-' '-' '-' '-'};
dot_arr = ['o' '*' '.' 'x' '+' 'V' '<' '>' '^'];

titlename = varargin{1};

%ref_ymin = 13407.77878;

%ref_ymin = 1779.246514; % Logistic Loss (C=10)
%ref_ymin_2 = 592.002541; % L2-Hinge Loss (C=10)

%ref_ymin = 3186.861213; % Logistic Loss (C=100)
%ref_ymin_2 = 622.248886; % L2-Hinge Loss (C=100)

%for i = 2:nargin  %([3 4 2 1]+1)
%		
%	filename = varargin{i};
%	fp = fopen(filename,'r');
%	line = fgets(fp);
%	
%	data = fscanf(fp,'%g',[2 inf]);
%	mini = min(data(2,:));
%	if( mini < ref_ymin )
%		ref_ymin = mini;
%	end
%	fclose(fp);
%end


all_names = titlename;
for i = 2:nargin  %([3 4 2 1]+1)
	
	
	filename = varargin{i};
	
	fp = fopen(filename,'r');
	line = fgets(fp);
	
	data = fscanf(fp,'%g',[2 inf]);
	legend('-DynamicLegend','Location','NorthEast');

	%if( i==2 )
		%data(2,:) = (data(2,:)-ref_ymin)/ref_ymin ;
	%else
	%	data(2,:) = (data(2,:)-ref_ymin_2)/ref_ymin_2;
	%end
	
	fname = split('/',filename);
	%if mod(i-1,2) ~= 0
	%	h1=semilogx(data(1,:),data(2,:),[line_arr{i-1} dot_arr{i-1} color_arr{i-1}],'DisplayName',fname{end},'LineWidth',1.0);
	%else
	%	h2=semilogx(data(1,:),data(2,:),[line_arr{i-1} dot_arr{i-1} color_arr{i-1}],'DisplayName',fname{end},'MarkerSize',15);	
	%	for j = 1:size(data,2)-1
	%		if( data(2,j) == data(2,j+1) )
	%			semilogx(data(1,j:j+1),data(2,j:j+1),'k-','LineWidth',2.0);
	%		end
	%	end
	%end
	semilogy(data(1,:),data(2,:),[line_arr{i-1} dot_arr{i-1} color_arr{i-1}], 'DisplayName', fname{end});
	%plot(data(1,:),data(2,:),[line_arr{i-1} dot_arr{i-1} color_arr{i-1}], 'DisplayName', fname{end});
	%loglog(data(1,:),data(2,:),[line_arr{i-1} dot_arr{i-1} color_arr{i-1}], 'DisplayName', fname{end});
	
	
	set(gca,'FontSize',18);
	hold on;
end
legend([h1 h2],'Location','NorthEast');

for i = 2:nargin
	filename = varargin{i};
	fp = fopen(filename,'r');
	line = fgets(fp);
	fname = split('/',filename);
	all_names = [all_names '_' fname{end}];
end

% plot level curve of \|w\|_1
%hbar = sum(abs(data(:,end)));
%plot( [hbar;0;-hbar;0;hbar], [0;hbar;0;-hbar;0], 'b-', 'DisplayName', 'level-curve');

% plot null-space at x(end)
%nvect = data(:,1)-data(:,2);
%pvect = [nvect(2);-nvect(1)];
%plot([data(1,end)-pvect(1) data(1,end)+pvect(1)],[data(2,end)-pvect(2) data(2,end)+pvect(2)], 'k-', 'DisplayName', 'z-dist=0');

filename = varargin{2};
fp = fopen(filename,'r');
line = fgets(fp);
labels = split(' ',line);
xlabel(labels{1});
ylabel(labels{2});
title(titlename);

axis([0,inf,0,inf]);
%axis equal;
%grid on;


%saveas(gcf,[all_names '.eps'],'epsc');
%saveas(gcf,[all_names '.pdf'],'pdf');
exit(0);
