function plotFig(varargin)

%color_arr = ['b' 'c' 'g' 'k' 'm' 'r' 'w' 'y' 'b'];
color_arr = { 'r'  'b' 'g' 'k' 'c'  'm'  'c'   'b'};
dot_arr = { ''  'o' 'o' 'x' 'x'  '*' 'x'    '^'};
%dot_arr = {'' '' '' '' '' '' '' ''    ''};
line_arr = {'-' '-' '-' '-' '-' '-' '-.' '-'    '-'};
%dot_arr = ['o' '*' '.' 'x' '+' 'V' '<' '>' '^'];

titlename = varargin{1};

%ref_ymin =  0.0;

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

	%data(2,:) = (data(2,:)-ref_ymin) ;
	
	%data = filter_data(data);
	%if(data(1,1)==0)
	%	data(1,1)=0.1;
	%end
	%data = data(:,1:200);

	fname = split('/',filename);
	
	%semilogx(data(1,:),data(2,:),[line_arr{i-1} dot_arr{i-1} color_arr{i-1}],'DisplayName',fname{end});
	%semilogy(data(1,:),data(2,:),[line_arr{i-1} dot_arr{i-1} color_arr{i-1}], 'DisplayName', fname{end});
	plot(data(1,:),data(2,:),[line_arr{i-1} dot_arr{i-1} color_arr{i-1}], 'DisplayName', fname{end});
	%loglog(data(1,:),data(2,:),[line_arr{i-1} dot_arr{i-1} color_arr{i-1}], 'DisplayName', fname{end});
	
	set(gca,'FontSize',18);
	hold on;
end
for i = 2:nargin
	filename = varargin{i};
	fp = fopen(filename,'r');
	line = fgets(fp);
	fname = split('/',filename);
	all_names = [all_names '_' fname{end}];
end

% plot box
a = 0;
b = 1.5;
plot([a b b a a],[a a b b a],'b-');

% plot null-space at x(end)
nvect = data(:,1)-data(:,2);
pvect = [nvect(2);-nvect(1)];
plot([data(1,end)-pvect(1) data(1,end)+pvect(1)],[data(2,end)-pvect(2) data(2,end)+pvect(2)], 'k-', 'DisplayName', 'z-dist=0');

filename = varargin{2};
fp = fopen(filename,'r');
line = fgets(fp);
labels = split(' ',line);
xlabel(labels{1});
ylabel(labels{2});


%axis([-inf,inf,-inf,inf]);
axis equal;
grid on;

title(titlename);

saveas(gcf,[all_names '.eps'],'epsc');
exit(0);
