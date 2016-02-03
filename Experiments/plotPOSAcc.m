
function plotPOSAcc(varargin)
is_acc = true;
color_arr = {'g' 'c' 'k' 'k' 'b' 'r' 'm' 'y' 'b'};
%color_arr = { 'r'  'k' 'r' 'r' 'k'  'k'  'c'   'b'};
dot_arr = { 'o' 'd'  '*'  's'  'x' 'x' 's'    '^'};
%dot_arr = {'' '' '' '' '' '' '' ''    ''};
line_arr = {'-' '-' '-' '-' '-' '-' '-' '-'    '-'};
%dot_arr = ['o' '*' '.' 'x' '+' 'V' '<' '>' '^'];

titlename = varargin{1};
xlower = -inf;
xupper = inf;
ylower = -inf;
%ylower = 0.0095;
%ylower = -inf;
yupper = inf;
%yupper = 0.4;
%yupper = 0.07;

%webspam-L2
%ref_ymin = 10090;

%webspam
%ref_ymin = 10127.329039;

%rcv1
%ref_ymin = 13407.77878;

%year
%ref_ymin = 1888284.0; %c=0.1
%ref_ymin = 196298.7079; %c=0.01

%e2006
%ref_ymin = 2565.1366;

%ref_ymin = 1779.246514; % Logistic Loss (C=10)
%ref_ymin_2 = 592.002541; % L2-Hinge Loss (C=10)

%ref_ymin = 3186.861213; % Logistic Loss (C=100)
%ref_ymin_2 = 622.248886; % L2-Hinge Loss (C=100)

ref_ymin = inf;
for i = 2:nargin  %([3 4 2 1]+1)
	
	filename = varargin{i};
	fp = fopen(filename,'r');
	line = fgets(fp);
	
	data = fscanf(fp,'%g',[2 inf]);
	if (size(data, 2) < 2)
		continue;
	end
	if (is_acc)
		data(2, :) = 1 - data(2, :);
	end
	for j = 2:size(data, 2)
		if data(2, j) > data(2, j-1)
			data(2, j) = data(2, j-1);
		end
	end
	mini = min(data(2,data(1,:)<=xupper));
	if( mini < ref_ymin )
		ref_ymin = mini;
	end
	fclose(fp);
end
ref_ymin = ref_ymin * 0.99;

all_names = titlename;
for i = 2:nargin  %([3 4 2 1]+1)
	
	
	filename = varargin{i};
	filename
	fp = fopen(filename,'r');
	line = fgets(fp);
	
	data = fscanf(fp,'%g',[2 inf]);  
	if (size(data, 2) < 2)
		continue;
	end
	if (is_acc)
		data(2, :) = 1 - data(2, :);
	end
	for j = 2:size(data, 2)
		if data(2, j) > data(2, j-1)
			data(2, j) = data(2, j-1);
		end
	end
	

	%legend('-DynamicLegend','Location','SouthEast');

	%data(2,:) = (data(2,:)-ref_ymin)/ref_ymin ;
	
	fname = split('/',filename);
	
	%semilogx(data(1,:),data(2,:),[line_arr{i-1} dot_arr{i-1} color_arr{i-1}],'DisplayName',fname{end});
	%semilogy(data(1,:),data(2,:),[line_arr{i-1} dot_arr{i-1} color_arr{i-1}], 'DisplayName', fname{end});
	plot(data(1,:),data(2,:),[line_arr{i-1} dot_arr{i-1} color_arr{i-1}], 'DisplayName', fname{end});
	%loglog(data(1,:),data(2,:),[line_arr{i-1} dot_arr{i-1} color_arr{i-1}], 'DisplayName', fname{end});
	
	set(gca,'FontSize',14);
	
	legend('-DynamicLegend','Location','NorthEast');
	hold on;
end

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
ylabel('test error');
title(titlename);

%axis([-inf.8,110,0,28]);
%axis([-inf,8000,0.1,0.4]);
xupper = 1000;
axis([xlower,xupper,ylower,yupper]);
%axis([-inf,inf,-inf,inf]);
%axis equal;
%grid on;


%saveas(gcf,[all_names '.eps'],'epsc');
saveas(gcf,[all_names '.pdf'],'pdf');
exit(0);
