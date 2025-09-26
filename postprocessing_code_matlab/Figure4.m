%% Generalized Agents
PathName1 = '/Volumes/Untitled/REACT_data/generalized/ws11/';
FileNameESP1 ='ESPData_series1.dat';
FileNameRT1 ='RTData_series1.dat';

PathName2 = '/Volumes/Untitled/REACT_data/generalized/ws12/';
FileNameESP2 ='ESPData_series1.dat';
FileNameRT2 ='RTData_series1.dat';

PathName3 = '/Volumes/Untitled/REACT_data/generalized/ws13/';
FileNameESP3 ='ESPData_series1.dat';
FileNameRT3 ='RTData_series1.dat';

PathName4 = '/Volumes/Untitled/REACT_data/generalized/ws15/';
FileNameESP4 ='ESPData_series1.dat';
FileNameRT4 ='RTData_series1.dat';

PathName5 = '/Volumes/Untitled/REACT_data/generalized/ws17/';
FileNameESP5 ='ESPData_series1.dat';
FileNameRT5 ='RTData_series1.dat';

PathName6 = '/Volumes/Untitled/REACT_data/generalized/ws18/';
FileNameESP6 ='ESPData_series1.dat';
FileNameRT6 ='RTData_series1.dat';

PathName7 = '/Volumes/Untitled/REACT_data/generalized/ws19/';
FileNameESP7 ='ESPData_series1.dat';
FileNameRT7 ='RTData_series1.dat';

PathName8 = '/Volumes/Untitled/REACT_data/generalized/ws20/';
FileNameESP8 ='ESPData_series1.dat';
FileNameRT8 ='RTData_series1.dat';
nchanESP = 64;
nchanRT = 26;
FsRT = 100;  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen([PathName1,FileNameESP1],'r','a');
DataESP1 = fread(fid,[nchanESP,inf],'float64');
fclose(fid);
[np_ESP,nt_ESP] = size(DataESP1); 
tESP = 0:1/FsRT:((nt_ESP-1)/FsRT);
disp('Loaded ESP1 data') 

fid = fopen([PathName1,FileNameRT1],'r','a');
DataRT1 = fread(fid,[nchanRT,inf],'float64');
fclose(fid);
[np_RT,nt_RT] = size(DataRT1);
tRT = 0:1/FsRT:((nt_RT-1)/FsRT);
disp('Loaded RT1 data')

step_start = find(DataRT1(23,:), 1, 'first');
step_end = size(DataRT1,2);
i_step = [step_start:step_end];
i_reset = [1: step_start-1];

DataESP1_eval_step_origin = DataESP1(:, i_step);
DataESP1_eval_reset_origin = DataESP1(:, i_reset);

DataRT1_eval_step_origin = DataRT1(:,i_step);
DataRT1_eval_reset_origin = DataRT1(:, i_reset);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen([PathName2,FileNameESP2],'r','a');
DataESP2 = fread(fid,[nchanESP,inf],'float64');
fclose(fid);
[np_ESP,nt_ESP] = size(DataESP2); 
tESP = 0:1/FsRT:((nt_ESP-1)/FsRT);
disp('Loaded ESP2 data')
fid = fopen([PathName2,FileNameRT2],'r','a');

DataRT2 = fread(fid,[nchanRT,inf],'float64');
fclose(fid);
[np_RT,nt_RT] = size(DataRT2);
tRT = 0:1/FsRT:((nt_RT-1)/FsRT);
disp('Loaded RT2 data')

step_start = find(DataRT2(23,:), 1, 'first');
step_end = size(DataRT2,2);
i_step = [step_start:step_end];
i_reset = [1: step_start-1];

DataESP2_eval_step_origin = DataESP2(:, i_step);
DataESP2_eval_reset_origin = DataESP2(:, i_reset);

DataRT2_eval_step_origin = DataRT2(:,i_step);
DataRT2_eval_reset_origin = DataRT2(:, i_reset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen([PathName3,FileNameESP3],'r','a');
DataESP3 = fread(fid,[nchanESP,inf],'float64');
fclose(fid);
[np_ESP,nt_ESP] = size(DataESP3); 
tESP = 0:1/FsRT:((nt_ESP-1)/FsRT);
disp('Loaded ESP3 data')

fid = fopen([PathName3,FileNameRT3],'r','a');
DataRT3 = fread(fid,[nchanRT,inf],'float64');
fclose(fid);
[np_RT,nt_RT] = size(DataRT3);
tRT = 0:1/FsRT:((nt_RT-1)/FsRT);
disp('Loaded RT3 data')


step_start = find(DataRT3(23,:), 1, 'first');
step_end = size(DataRT3,2);
i_step = [step_start:step_end];
i_reset = [1: step_start-1];

DataESP3_eval_step_origin = DataESP3(:, i_step);
DataESP3_eval_reset_origin = DataESP3(:, i_reset);

DataRT3_eval_step_origin = DataRT3(:,i_step);
DataRT3_eval_reset_origin = DataRT3(:, i_reset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen([PathName4,FileNameESP4],'r','a');
DataESP4 = fread(fid,[nchanESP,inf],'float64');
fclose(fid);
[np_ESP,nt_ESP] = size(DataESP4); 
tESP = 0:1/FsRT:((nt_ESP-1)/FsRT);
disp('Loaded ESP4 data')

fid = fopen([PathName4,FileNameRT4],'r','a');
DataRT4 = fread(fid,[nchanRT,inf],'float64');
fclose(fid);
[np_RT,nt_RT] = size(DataRT4);
tRT = 0:1/FsRT:((nt_RT-1)/FsRT);
disp('Loaded RT4 data')

step_start = find(DataRT4(23,:), 1, 'first');
step_end = size(DataRT4,2);
i_step = [step_start:step_end];
i_reset = [1: step_start-1];

DataESP4_eval_step_origin = DataESP4(:, i_step);
DataESP4_eval_reset_origin = DataESP4(:, i_reset);

DataRT4_eval_step_origin = DataRT4(:,i_step);
DataRT4_eval_reset_origin = DataRT4(:, i_reset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen([PathName5,FileNameESP5],'r','a');
DataESP5 = fread(fid,[nchanESP,inf],'float64');
fclose(fid);
[np_ESP,nt_ESP] = size(DataESP5); 
tESP = 0:1/FsRT:((nt_ESP-1)/FsRT);
disp('Loaded ESP5 data')

fid = fopen([PathName5,FileNameRT5],'r','a');
DataRT5 = fread(fid,[nchanRT,inf],'float64');
fclose(fid);
[np_RT,nt_RT] = size(DataRT5);
tRT = 0:1/FsRT:((nt_RT-1)/FsRT);
disp('Loaded RT5 data')

step_start = find(DataRT5(23,:), 1, 'first');
step_end = size(DataRT5,2);
i_step = [step_start:step_end];
i_reset = [1: step_start-1];

DataESP5_eval_step_origin = DataESP5(:, i_step);
DataESP5_eval_reset_origin = DataESP5(:, i_reset);

DataRT5_eval_step_origin = DataRT5(:,i_step);
DataRT5_eval_reset_origin = DataRT5(:, i_reset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen([PathName6,FileNameESP6],'r','a');
DataESP6 = fread(fid,[nchanESP,inf],'float64');
fclose(fid);
[np_ESP,nt_ESP] = size(DataESP6); 
tESP = 0:1/FsRT:((nt_ESP-1)/FsRT);
disp('Loaded ESP6 data')

fid = fopen([PathName6,FileNameRT6],'r','a');
DataRT6 = fread(fid,[nchanRT,inf],'float64');
fclose(fid);
[np_RT,nt_RT] = size(DataRT6);
tRT = 0:1/FsRT:((nt_RT-1)/FsRT);
disp('Loaded RT6 data')

step_start = find(DataRT6(23,:), 1, 'first');
step_end = size(DataRT6,2);
i_step = [step_start:step_end];
i_reset = [1: step_start-1];

DataESP6_eval_step_origin = DataESP6(:, i_step);
DataESP6_eval_reset_origin = DataESP6(:, i_reset);

DataRT6_eval_step_origin = DataRT6(:,i_step);
DataRT6_eval_reset_origin = DataRT6(:, i_reset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen([PathName7,FileNameESP7],'r','a');
DataESP7 = fread(fid,[nchanESP,inf],'float64');
fclose(fid);
[np_ESP,nt_ESP] = size(DataESP7); 
tESP = 0:1/FsRT:((nt_ESP-1)/FsRT);
disp('Loaded ESP7 data')

fid = fopen([PathName7,FileNameRT7],'r','a');
DataRT7 = fread(fid,[nchanRT,inf],'float64');
fclose(fid);
[np_RT,nt_RT] = size(DataRT7);
tRT = 0:1/FsRT:((nt_RT-1)/FsRT);
disp('Loaded RT7 data')

step_start = find(DataRT7(23,:), 1, 'first');
step_end = size(DataRT7,2);
i_step = [step_start:step_end];
i_reset = [1: step_start-1];

DataESP7_eval_step_origin = DataESP7(:, i_step);
DataESP7_eval_reset_origin = DataESP7(:, i_reset);

DataRT7_eval_step_origin = DataRT7(:,i_step);
DataRT7_eval_reset_origin = DataRT7(:, i_reset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen([PathName8,FileNameESP8],'r','a');
DataESP8 = fread(fid,[nchanESP,inf],'float64');
fclose(fid);
[np_ESP,nt_ESP] = size(DataESP8); 
tESP = 0:1/FsRT:((nt_ESP-1)/FsRT);
disp('Loaded ESP8 data')

fid = fopen([PathName8,FileNameRT8],'r','a');
DataRT8 = fread(fid,[nchanRT,inf],'float64');
fclose(fid);
[np_RT,nt_RT] = size(DataRT8);
tRT = 0:1/FsRT:((nt_RT-1)/FsRT);
disp('Loaded RT8 data')

step_start = find(DataRT8(23,:), 1, 'first');
step_end = size(DataRT8,2);
i_step = [step_start:step_end];
i_reset = [1: step_start-1];

DataESP8_eval_step_origin = DataESP8(:, i_step);
DataESP8_eval_reset_origin = DataESP8(:, i_reset);

DataRT8_eval_step_origin = DataRT8(:,i_step);
DataRT8_eval_reset_origin = DataRT8(:, i_reset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Specialized Agents
PathName1 = '/Volumes/Untitled/REACT_data/specialized/ws11/';
FileNameESP1 ='ESPData_series1.dat';
FileNameRT1 ='RTData_series1.dat';

PathName2 = '/Volumes/Untitled/REACT_data/specialized/ws12/';
FileNameESP2 ='ESPData_series1.dat';
FileNameRT2 ='RTData_series1.dat';

PathName3 = '/Volumes/Untitled/REACT_data/specialized/ws13/';
FileNameESP3 ='ESPData_series1.dat';
FileNameRT3 ='RTData_series1.dat';

PathName4 = '/Volumes/Untitled/REACT_data/specialized/ws15/';
FileNameESP4 ='ESPData_series1.dat';
FileNameRT4 ='RTData_series1.dat';

PathName5 = '/Volumes/Untitled/REACT_data/specialized/ws17/';
FileNameESP5 ='ESPData_series1.dat';
FileNameRT5 ='RTData_series1.dat';

PathName6 = '/Volumes/Untitled/REACT_data/specialized/ws18/';
FileNameESP6 ='ESPData_series1.dat';
FileNameRT6 ='RTData_series1.dat';

PathName7 = '/Volumes/Untitled/REACT_data/specialized/ws19/';
FileNameESP7 ='ESPData_series1.dat';
FileNameRT7 ='RTData_series1.dat';

PathName8 = '/Volumes/Untitled/REACT_data/specialized/ws20/';
FileNameESP8 ='ESPData_series1.dat';
FileNameRT8 ='RTData_series1.dat';
nchanESP = 64;
nchanRT = 26;
FsRT = 100;  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen([PathName1,FileNameESP1],'r','a');
DataESP1 = fread(fid,[nchanESP,inf],'float64');
fclose(fid);
[np_ESP,nt_ESP] = size(DataESP1); 
tESP = 0:1/FsRT:((nt_ESP-1)/FsRT);
disp('Loaded ESP1 data') 

fid = fopen([PathName1,FileNameRT1],'r','a');
DataRT1 = fread(fid,[nchanRT,inf],'float64');
fclose(fid);
[np_RT,nt_RT] = size(DataRT1);
tRT = 0:1/FsRT:((nt_RT-1)/FsRT);
disp('Loaded RT1 data')

step_start = find(DataRT1(23,:), 1, 'first');
step_end = size(DataRT1,2);
i_step = [step_start:step_end];
i_reset = [1: step_start-1];

DataESP1_eval_step_basic = DataESP1(:, i_step);
DataESP1_eval_reset_basic = DataESP1(:, i_reset);

DataRT1_eval_step_basic = DataRT1(:,i_step);
DataRT1_eval_reset_basic = DataRT1(:, i_reset);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen([PathName2,FileNameESP2],'r','a');
DataESP2 = fread(fid,[nchanESP,inf],'float64');
fclose(fid);
[np_ESP,nt_ESP] = size(DataESP2); 
tESP = 0:1/FsRT:((nt_ESP-1)/FsRT);
disp('Loaded ESP2 data')
fid = fopen([PathName2,FileNameRT2],'r','a');

DataRT2 = fread(fid,[nchanRT,inf],'float64');
fclose(fid);
[np_RT,nt_RT] = size(DataRT2);
tRT = 0:1/FsRT:((nt_RT-1)/FsRT);
disp('Loaded RT2 data')

step_start = find(DataRT2(23,:), 1, 'first');
step_end = size(DataRT2,2);
i_step = [step_start:step_end];
i_reset = [1: step_start-1];

DataESP2_eval_step_basic = DataESP2(:, i_step);
DataESP2_eval_reset_basic = DataESP2(:, i_reset);

DataRT2_eval_step_basic = DataRT2(:,i_step);
DataRT2_eval_reset_basic = DataRT2(:, i_reset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen([PathName3,FileNameESP3],'r','a');
DataESP3 = fread(fid,[nchanESP,inf],'float64');
fclose(fid);
[np_ESP,nt_ESP] = size(DataESP3); 
tESP = 0:1/FsRT:((nt_ESP-1)/FsRT);
disp('Loaded ESP3 data')

fid = fopen([PathName3,FileNameRT3],'r','a');
DataRT3 = fread(fid,[nchanRT,inf],'float64');
fclose(fid);
[np_RT,nt_RT] = size(DataRT3);
tRT = 0:1/FsRT:((nt_RT-1)/FsRT);
disp('Loaded RT3 data')


step_start = find(DataRT3(23,:), 1, 'first');
step_end = size(DataRT3,2);
i_step = [step_start:step_end];
i_reset = [1: step_start-1];

DataESP3_eval_step_basic = DataESP3(:, i_step);
DataESP3_eval_reset_basic = DataESP3(:, i_reset);

DataRT3_eval_step_basic = DataRT3(:,i_step);
DataRT3_eval_reset_basic = DataRT3(:, i_reset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen([PathName4,FileNameESP4],'r','a');
DataESP4 = fread(fid,[nchanESP,inf],'float64');
fclose(fid);
[np_ESP,nt_ESP] = size(DataESP4); 
tESP = 0:1/FsRT:((nt_ESP-1)/FsRT);
disp('Loaded ESP4 data')

fid = fopen([PathName4,FileNameRT4],'r','a');
DataRT4 = fread(fid,[nchanRT,inf],'float64');
fclose(fid);
[np_RT,nt_RT] = size(DataRT4);
tRT = 0:1/FsRT:((nt_RT-1)/FsRT);
disp('Loaded RT4 data')

step_start = find(DataRT4(23,:), 1, 'first');
step_end = size(DataRT4,2);
i_step = [step_start:step_end];
i_reset = [1: step_start-1];

DataESP4_eval_step_basic = DataESP4(:, i_step);
DataESP4_eval_reset_basic = DataESP4(:, i_reset);

DataRT4_eval_step_basic = DataRT4(:,i_step);
DataRT4_eval_reset_basic = DataRT4(:, i_reset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen([PathName5,FileNameESP5],'r','a');
DataESP5 = fread(fid,[nchanESP,inf],'float64');
fclose(fid);
[np_ESP,nt_ESP] = size(DataESP5); 
tESP = 0:1/FsRT:((nt_ESP-1)/FsRT);
disp('Loaded ESP5 data')

fid = fopen([PathName5,FileNameRT5],'r','a');
DataRT5 = fread(fid,[nchanRT,inf],'float64');
fclose(fid);
[np_RT,nt_RT] = size(DataRT5);
tRT = 0:1/FsRT:((nt_RT-1)/FsRT);
disp('Loaded RT5 data')

step_start = find(DataRT5(23,:), 1, 'first');
step_end = size(DataRT5,2);
i_step = [step_start:step_end];
i_reset = [1: step_start-1];

DataESP5_eval_step_basic = DataESP5(:, i_step);
DataESP5_eval_reset_basic = DataESP5(:, i_reset);

DataRT5_eval_step_basic = DataRT5(:,i_step);
DataRT5_eval_reset_basic = DataRT5(:, i_reset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen([PathName6,FileNameESP6],'r','a');
DataESP6 = fread(fid,[nchanESP,inf],'float64');
fclose(fid);
[np_ESP,nt_ESP] = size(DataESP6); 
tESP = 0:1/FsRT:((nt_ESP-1)/FsRT);
disp('Loaded ESP6 data')

fid = fopen([PathName6,FileNameRT6],'r','a');
DataRT6 = fread(fid,[nchanRT,inf],'float64');
fclose(fid);
[np_RT,nt_RT] = size(DataRT6);
tRT = 0:1/FsRT:((nt_RT-1)/FsRT);
disp('Loaded RT6 data')

step_start = find(DataRT6(23,:), 1, 'first');
step_end = size(DataRT6,2);
i_step = [step_start:step_end];
i_reset = [1: step_start-1];

DataESP6_eval_step_basic = DataESP6(:, i_step);
DataESP6_eval_reset_basic = DataESP6(:, i_reset);

DataRT6_eval_step_basic = DataRT6(:,i_step);
DataRT6_eval_reset_basic = DataRT6(:, i_reset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen([PathName7,FileNameESP7],'r','a');
DataESP7 = fread(fid,[nchanESP,inf],'float64');
fclose(fid);
[np_ESP,nt_ESP] = size(DataESP7); 
tESP = 0:1/FsRT:((nt_ESP-1)/FsRT);
disp('Loaded ESP7 data')

fid = fopen([PathName7,FileNameRT7],'r','a');
DataRT7 = fread(fid,[nchanRT,inf],'float64');
fclose(fid);
[np_RT,nt_RT] = size(DataRT7);
tRT = 0:1/FsRT:((nt_RT-1)/FsRT);
disp('Loaded RT7 data')

step_start = find(DataRT7(23,:), 1, 'first');
step_end = size(DataRT7,2);
i_step = [step_start:step_end];
i_reset = [1: step_start-1];

DataESP7_eval_step_basic = DataESP7(:, i_step);
DataESP7_eval_reset_basic = DataESP7(:, i_reset);

DataRT7_eval_step_basic = DataRT7(:,i_step);
DataRT7_eval_reset_basic = DataRT7(:, i_reset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen([PathName8,FileNameESP8],'r','a');
DataESP8 = fread(fid,[nchanESP,inf],'float64');
fclose(fid);
[np_ESP,nt_ESP] = size(DataESP8); 
tESP = 0:1/FsRT:((nt_ESP-1)/FsRT);
disp('Loaded ESP8 data')

fid = fopen([PathName8,FileNameRT8],'r','a');
DataRT8 = fread(fid,[nchanRT,inf],'float64');
fclose(fid);
[np_RT,nt_RT] = size(DataRT8);
tRT = 0:1/FsRT:((nt_RT-1)/FsRT);
disp('Loaded RT8 data')

step_start = find(DataRT8(23,:), 1, 'first');
step_end = size(DataRT8,2);
i_step = [step_start:step_end];
i_reset = [1: step_start-1];

DataESP8_eval_step_basic = DataESP8(:, i_step);
DataESP8_eval_reset_basic = DataESP8(:, i_reset);

DataRT8_eval_step_basic = DataRT8(:,i_step);
DataRT8_eval_reset_basic = DataRT8(:, i_reset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 
DataESP_step_Series = {DataESP1_eval_step_origin, DataESP2_eval_step_origin, DataESP3_eval_step_origin, DataESP4_eval_step_origin,...
                DataESP5_eval_step_origin, DataESP6_eval_step_origin, DataESP7_eval_step_origin, DataESP8_eval_step_origin}

DataESP_reset_Series = {DataESP1_eval_reset_origin, DataESP2_eval_reset_origin, DataESP3_eval_reset_origin, DataESP4_eval_reset_origin,...
                DataESP5_eval_reset_origin, DataESP6_eval_reset_origin, DataESP7_eval_reset_origin, DataESP8_eval_reset_origin}

DataESP_step_Series_basic = {DataESP1_eval_step_basic, DataESP2_eval_step_basic, DataESP3_eval_step_basic, DataESP4_eval_step_basic,...
                DataESP5_eval_step_basic, DataESP6_eval_step_basic, DataESP7_eval_step_basic, DataESP8_eval_step_basic}

DataESP_reset_Series_basic = {DataESP1_eval_reset_basic, DataESP2_eval_reset_basic, DataESP3_eval_reset_basic, DataESP4_eval_reset_basic,...
                DataESP5_eval_reset_basic, DataESP6_eval_reset_basic, DataESP7_eval_reset_basic, DataESP8_eval_reset_basic}

DataRT_step_Series = {DataRT1_eval_step_origin, DataRT2_eval_step_origin, DataRT3_eval_step_origin, DataRT4_eval_step_origin,...
                DataRT5_eval_step_origin, DataRT6_eval_step_origin, DataRT7_eval_step_origin, DataRT8_eval_step_origin}

DataRT_reset_Series = {DataRT1_eval_reset_origin, DataRT2_eval_reset_origin, DataRT3_eval_reset_origin, DataRT4_eval_reset_origin,...
                DataRT5_eval_reset_origin, DataRT6_eval_reset_origin, DataRT7_eval_reset_origin, DataRT8_eval_reset_origin}

DataRT_step_Series_basic = {DataRT1_eval_step_basic, DataRT2_eval_step_basic, DataRT3_eval_step_basic, DataRT4_eval_step_basic,...
                DataRT5_eval_step_basic, DataRT6_eval_step_basic, DataRT7_eval_step_basic, DataRT8_eval_step_basic}

DataRT_reset_Series_basic = {DataRT1_eval_reset_basic, DataRT2_eval_reset_basic, DataRT3_eval_reset_basic, DataRT4_eval_reset_basic,...
                DataRT5_eval_reset_basic, DataRT6_eval_reset_basic, DataRT7_eval_reset_basic, DataRT8_eval_reset_basic}



origin_dragreduction_list = [];
basic_dragreduction_list = [];


origin_pressure_list = [];
basic_pressure_list = [];
uncontrolled_pressure_list = [];


for i = 1:8

    origin_dragreduction_list = [origin_dragreduction_list  dragreductioncalculator(DataRT_step_Series{i}, DataRT_reset_Series{i})];
    basic_dragreduction_list = [basic_dragreduction_list  dragreductioncalculator(DataRT_step_Series_basic{i}, DataRT_reset_Series_basic{i})];

    origin_pressure_list = [origin_pressure_list, pressuremeancalculator(DataESP_step_Series{i})];

    basic_pressure_list = [basic_pressure_list, pressuremeancalculator(DataESP_step_Series_basic{i})];

    uncontrolled_pressure_list = [uncontrolled_pressure_list, pressuremeancalculator(DataESP_reset_Series{i})];

end

%% Drag Reduction Plot

figure('Position', [100, 100, 680, 500]); 
hold on;
wind_class = [11,12,13,15,17,18,19,20];

p1 = plot(wind_class, origin_dragreduction_list*100, '-o', 'LineWidth', 3.3, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.2549, 0.4118, 0.8824], 'Color', [0.2549, 0.4118, 0.8824], 'DisplayName', 'Generalized Controller');

p2 = plot(wind_class, basic_dragreduction_list*100, '-s', 'LineWidth', 3.3, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.5 0.5 0.5], 'Color', [0.5 0.5 0.5], 'DisplayName', 'Specialized Controller');


xlim([min(wind_class)-1, max(wind_class)+1]);
% Labels and title
xlabel('Vehicle speed (m/s)', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('Drag reduction (\%)', 'FontSize', 14, 'Interpreter', 'latex');

% Improve axis ticks and font
set(gca, 'TickLabelInterpreter', 'latex');
set(gca, 'FontSize', 23, 'LineWidth', 1.6, 'TickDir', 'out');


% Add legend
legend('Location', 'southeast', 'FontSize', 20, 'Box', 'off');

% Adjust figure appearance
set(gcf, 'Color', 'w'); % White background

%% Pressure Mean
figure('Position', [100, 100, 680, 500]);  % 800x600 pixels figure
hold on;

wind_class = [11,12,13,15,17,18,19,20];

p1 = plot(wind_class, uncontrolled_pressure_list, '-o', 'LineWidth', 3, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0, 0, 0], 'Color', [0, 0, 0], 'DisplayName', 'Uncontrolled');


p2 = plot(wind_class, origin_pressure_list, '--s', 'LineWidth', 3, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.2549, 0.4118, 0.8824], 'Color', [0.2549, 0.4118, 0.8824], 'DisplayName', 'Generalized Controller');


xlim([min(wind_class)-1, max(wind_class)+1]);
ylim([-33, -8])
% Labels and title
xlabel('Vehicle speed (m/s)', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('Mean base pressure (Pa)', 'FontSize', 14, 'Interpreter', 'latex');

% Improve axis ticks and font

set(gca, 'TickLabelInterpreter', 'latex');
set(gca, 'FontSize', 23, 'LineWidth', 1.6, 'TickDir', 'out');


% Add legend
legend('Location', 'southwest', 'FontSize', 20, 'Box', 'off');

% Adjust figure appearance
set(gcf, 'Color', 'w'); % White background

%% Controller Trajectory Specialized General Comparison 

Step_C1_list = [];
Step_A1_list = [];

Step_C1_basic_list = [];
Step_A1_basic_list = [];
wind_class = [11,12,13,15,17,18,19,20];
line_styles = {'-', '--', ':', '-.', '-', '--', ':', '-.'};  % Define line styles for 8 series
nwindow = 2^10
for i = 1:8

    DataRT_step_A = DataRT_step_Series{i};

    [pxxQ26, f_PSDQ26] = pwelch(detrend(DataRT_step_A(23,:)), hanning(nwindow), 0.5*nwindow, nwindow, FsRT);

    [~, idx] = max(pxxQ26.*f_PSDQ26); 
    peak_frequency = f_PSDQ26(idx); 
    Step_C1_list = [Step_C1_list, peak_frequency];
    Step_A1_list = [Step_A1_list, max(pxxQ26.*f_PSDQ26)];

    DataRT_step_A_basic = DataRT_step_Series_basic{i};
    
    [pxxQ26, f_PSDQ26] = pwelch(detrend(DataRT_step_A_basic(23,:)), hanning(nwindow), 0.5*nwindow, nwindow, FsRT);

    [~, idx] = max(pxxQ26.*f_PSDQ26); 
    peak_frequency = f_PSDQ26(idx); 
    Step_C1_basic_list = [Step_C1_basic_list, peak_frequency];
    Step_A1_basic_list = [Step_A1_basic_list, max(pxxQ26.*f_PSDQ26)];




end




figure
hold on

% Data
wind_class = [11,12,13,15,17,18,19,20];
wind_class_basic = wind_class(2:8);
Step_C1_basic_list = Step_C1_basic_list(2:8);
Step_A1_basic_list = Step_A1_basic_list(2:8);

% Interpolated points
wind_interp = linspace(min(wind_class), max(wind_class), 100);
wind_interp_basic = linspace(min(wind_class_basic), max(wind_class_basic), 100);

C1_interp_basic = interp1(wind_class_basic, Step_C1_basic_list, wind_interp_basic, 'pchip');
B1_interp_basic = interp1(wind_class_basic, Step_A1_basic_list, wind_interp_basic, 'pchip');

C1_interp = interp1(wind_class, Step_C1_list, wind_interp, 'pchip');
B1_interp = interp1(wind_class, Step_A1_list, wind_interp, 'pchip');

% Scatter plots
scatter3(wind_class, Step_C1_list, Step_A1_list, ...
    100, [0.2549, 0.4118, 0.8824], 'filled', 'MarkerEdgeColor', [0.2549, 0.4118, 0.8824], 'LineWidth', 1.5, 'MarkerFaceAlpha', 0.6);

scatter3(wind_class_basic, Step_C1_basic_list, Step_A1_basic_list, ...
    100, [0.5 0.5 0.5], 'filled', 'MarkerEdgeColor', [0.5 0.5 0.5], 'LineWidth', 1.5, 'MarkerFaceAlpha', 0.6);



% Interpolated lines
plot3(wind_interp, C1_interp, B1_interp, 'Color', [0.2549, 0.4118, 0.8824] , 'LineWidth', 3);                    % Controlled fit
plot3(wind_interp_basic, C1_interp_basic, B1_interp_basic,  'Color', [0.5 0.5 0.5], 'LineWidth', 3);   % Baseline fit


ax = gca;
ax.FontSize = 16;
% Labels

xl = xlabel('Vehicle speed (m/s)', 'FontSize', 18, 'Rotation', 40, 'Interpreter', 'latex');
yl = ylabel('$f_{\mathrm{dominant}}$ (Hz)', 'Interpreter', 'latex', 'FontSize', 23, 'Rotation', -15);
zl = zlabel('$\Phi(f_{\mathrm{dominant}})$', 'Interpreter', 'latex', 'FontSize', 23, 'Rotation', 90);
%xl.Position(3) = xl.Position(3) - 0;
yl.Position(1) = yl.Position(1) - 1;  % Shift along x
disp(yl.Position);



% Axis limits
xlim([11,21]);
ylim([0.7, 1.8]);
ylim([0.7, 3]);

% Font and grid

grid minor;

% Background color
set(gca, 'Color', [0.95 0.95 0.95]);

% View and lighting
view(301.6098, 48.8688);        % Adjust for better Z trend visibility
lighting phong;
camlight('headlight');
material shiny;

% Legend
legend({'Generalized Controller', 'Specialized Controller'}, ...
       'FontSize', 16, 'Location', 'northeast');

hold off;
% Font and grid

grid minor;
ax.Box = 'on';
ax.BoxStyle = 'full';




