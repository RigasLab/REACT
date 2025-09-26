PathName = '/Volumes/Untitled/REACT_data/generalized/';
FileNameESP ='ESPData_series1_100s.dat';
FileNameRT ='RTData_series1_100s.dat';
graph_saving_path = [PathName 'graph_produced/']

graph_saving_path2 = [PathName 'graph_produced2/']

nchanESP = 64;
nchanRT = 26;

FsRT = 100;              % Sampling frequency [Hz]
Uinf = 20;             % Freestream velocity [m/s]
fscale      = 0.216/15; % Nondimensionlize frequency to St, with respect to the width of the body
fscale_z    = 0.160/15; % Nondimensionlize with the height of the body
nwindow     = 2^11; % Length of the window used in pwelch 

rho = 1.204;            % air density at 20C
q   = 0.5*rho*Uinf^2;   % dynamic head


%% Read ESP (64 taps) 

fid = fopen([PathName,FileNameESP],'r','a');
DataESP = fread(fid,[nchanESP,inf],'float64');
fclose(fid);

[np_ESP,nt_ESP] = size(DataESP); 
tESP = 0:1/FsRT:((nt_ESP-1)/FsRT);

disp('Loaded ESP data')


%% Read RT

fid = fopen([PathName,FileNameRT],'r','a');
DataRT = fread(fid,[nchanRT,inf],'float64');
fclose(fid);

[np_RT,nt_RT] = size(DataRT);
tRT = 0:1/FsRT:((nt_RT-1)/FsRT);

disp('Loaded RT data')


%% 

i_step = [10003:20002]; % Convert selected points to indices
i_reset = 3:10002;
DataESP_eval_step = DataESP(:, i_step);
DataESP_eval_reset = DataESP(:, i_reset);

DataRT_eval_step = DataRT(:,i_step);
DataRT_eval_reset = DataRT(:, i_reset);

DataESP_cf_selection = DataESP_eval_step;
DataESP_cf_reset_selection = DataESP_eval_reset;

DataRT_cf_reset_selection = DataRT_eval_reset;
DataRT_cf_selection = DataRT_eval_step;
range_step = i_step;
range_reset = i_reset;



%% Time Series


% CoP (x is width, y is height)
[CoPx_cf2, ~]       = EvalCoP(DataESP_cf_selection);
CoPx_cf2(isnan(CoPx_cf2)) = 0;

[CoPx_cf2_reset, ~] = EvalCoP(DataESP_cf_reset_selection);
CoPx_cf2_reset(isnan(CoPx_cf2_reset)) = 0;

% Time vectors
time_vector          = (1:20000) / FsRT;
time_vector_section  = (10001:20000) / FsRT;

% Colors
grey_color = [0.65, 0.65, 0.65];
pink_color = [0.95, 0.55, 0.65];

% Combine data
CoPx_combine   = [CoPx_cf2_reset, CoPx_cf2];                          % Blue lines (raw)
Drag_combine   = [-movmean(DataRT_cf_reset_selection(8,:), 60), ...
                  -movmean(DataRT_cf_selection(8,:), 60)];            % Drag (moving mean)
CoPx_section   = movmean(CoPx_cf2, 100);                              % Red moving mean line
Action_combine = [DataRT_cf_reset_selection(24,:), ...
                  DataRT_cf_selection(24,:)];
Action_combine2 = [DataRT_cf_reset_selection(23,:), ...
                   DataRT_cf_selection(23,:)];
Pressure_combine = [movmean(mean(DataESP_cf_reset_selection(1:64,:),1), 30), ...
                    movmean(mean(DataESP_cf_selection(1:64,:),1), 30)];

% Energy saved
Power_saved          = (mean(Drag_combine(1:10000)) - (-DataRT_cf_selection(8, :))) * 15 ...
                       - (DataRT_cf_selection(11, :) + DataRT_cf_selection(12, :));
Power_saved_cumsum   = cumsum(Power_saved);
Power_combine        = [zeros(1, 10000), Power_saved_cumsum] / 100;

% Figure setup
figure('Color', 'w', 'Units', 'pixels', 'Position', [100, 100, 800, 800]);
tiledlayout(5, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
fontSize = 14;

% --- Actions ---
nexttile; hold on;
h1 = plot(time_vector, Action_combine(1,:),  'Color', pink_color, 'LineWidth', 0.5);
h3 = plot(time_vector, Action_combine2(1,:), 'Color', grey_color, 'LineWidth', 0.5);
yl = ylim;
h2 = fill([100, max(time_vector), max(time_vector), 100], ...
          [yl(1), yl(1), yl(2), yl(2)], ...
          [0.8, 1.0, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
uistack(h2, 'bottom');
ylabel('Control signal', 'FontSize', fontSize);
legend([h2, h1, h3], {'RL Controller Activated', 'Left flap', 'Right flap'}, ...
       'FontSize', fontSize, 'Box', 'off', 'Location', 'north');
set(gca, 'FontSize', fontSize, 'TickDir', 'out', 'Box', 'off');

% --- Drag ---
nexttile; hold on;
plot(time_vector, Drag_combine(1,:), 'Color', [0.2549, 0.4118, 0.8824], 'LineWidth', 1.5);
yl = ylim;
h2 = fill([100, max(time_vector), max(time_vector), 100], ...
          [yl(1), yl(1), yl(2), yl(2)], ...
          [0.8, 1.0, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
uistack(h2, 'bottom');
ylabel('Drag (N)', 'FontSize', fontSize);
set(gca, 'FontSize', fontSize, 'TickDir', 'out', 'Box', 'off');

% --- Base pressure ---
nexttile; hold on;
plot(time_vector, Pressure_combine(1,:), 'Color', [0.2549, 0.4118, 0.8824], 'LineWidth', 1.5);
yl = ylim;
h2 = fill([100, max(time_vector), max(time_vector), 100], ...
          [yl(1), yl(1), yl(2), yl(2)], ...
          [0.8, 1.0, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
uistack(h2, 'bottom');
ylabel('Base pressure (Pa)', 'FontSize', fontSize);
set(gca, 'FontSize', fontSize, 'TickDir', 'out', 'Box', 'off');

% --- CoP_x ---
nexttile; hold on;
plot(time_vector, CoPx_combine(1,:), 'Color', [0.2549, 0.4118, 0.8824], 'LineWidth', 1.5);
plot(time_vector_section, CoPx_section, 'Color', 'red', 'LineWidth', 1.5);
yl = ylim;
h2 = fill([100, max(time_vector), max(time_vector), 100], ...
          [yl(1), yl(1), yl(2), yl(2)], ...
          [0.8, 1.0, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
uistack(h2, 'bottom');
ylabel('$CoP_z$', 'Interpreter', 'latex', 'FontSize', fontSize);
set(gca, 'FontSize', fontSize, 'TickDir', 'out', 'Box', 'off');

% --- Energy saved ---
nexttile; hold on;
plot(time_vector, Power_combine(1,:), 'Color', [0.2549, 0.4118, 0.8824], 'LineWidth', 3);
yl = ylim;
h2 = fill([100, max(time_vector), max(time_vector), 100], ...
          [yl(1), yl(1), yl(2), yl(2)], ...
          [0.8, 1.0, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
uistack(h2, 'bottom');
ylabel('Energy saved (J)', 'FontSize', fontSize);
set(gca, 'FontSize', fontSize, 'TickDir', 'out', 'Box', 'off');
xlabel('Time (s)', 'FontSize', fontSize);




%% Pressure Contour
% Grid and resolution
Nx   = 8;
Ny   = 8;
resX = 20;
resY = 20;

% Data aliases
data_contour_step  = DataESP_eval_step;
data_contour_reset = DataESP_eval_reset;

% --- No-control mean ---
figure; clf; hold all;
set(gca, 'LooseInset', [0, 0, 0, 0]);              % Remove extra padding
plotESP_Ahmed(mean(data_contour_reset, 2), Nx, Ny, resX, resY);
xlabel("Z");
ylabel("Y");
colorbar();
colormap('bluewhitered');
title('No control');
set(gca, 'OuterPosition', [0, 0, 1, 1]);           % Maximize figure usage
set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.02)); % Tight layout

% --- Controlled mean ---
figure; clf; hold all;
set(gca, 'LooseInset', [0, 0, 0, 0]);
plotESP_Ahmed(mean(data_contour_step, 2), Nx, Ny, resX, resY);
xlabel("Y");
ylabel("Z");
colorbar();
colormap('bluewhitered');
title('Control');
set(gca, 'OuterPosition', [0, 0, 1, 1]);
set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.02));

% --- Percentage difference ---
data_contour_change = 100 * ...
    (mean(data_contour_step,  2) - mean(data_contour_reset, 2)) ./ ...
     abs(mean(data_contour_reset, 2));

figure; hold all;
set(gca, 'LooseInset', [0, 0, 0, 0]);

% Higher-resolution rendering for the change plot
resX = 200;
resY = 200;

plotESP_Ahmed(data_contour_change, Nx, Ny, resX, resY);
xlabel('$Z\,(\mathrm{mm})$', 'Interpreter', 'latex');
ylabel('$Y\,(\mathrm{mm})$', 'Interpreter', 'latex');

% Colorbar and labels
hcb = colorbar();
caxis([-25 25]);
tick_vals   = hcb.Ticks;
tick_labels = strcat('$', string(tick_vals), '\%$');
set(hcb, 'Ticks', tick_vals, ...
         'TickLabels', tick_labels, ...
         'TickLabelInterpreter', 'latex');

% Center white in a custom bluewhitered colormap
num_colors       = 256;
custom_colormap  = bluewhitered(num_colors);
middle_index     = round(num_colors / 2);
custom_colormap(middle_index, :) = [1 1 1];
colormap(custom_colormap);

title('Time-averaged base pressure recovery (\%)', ...
      'Interpreter', 'latex', 'FontWeight', 'normal');

set(gca, 'FontSize', 16);
set(gca, 'OuterPosition', [0, 0, 1, 1]);
set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.050));


%% RMS Contour

% Controlled RMS
figure; clf; hold all;
plotESP_Ahmed(std(data_contour_step, 0, 2), Nx, Ny, resX, resY);
xlabel("Y");
ylabel("Z");
colorbar();
title('Controlled RMS');

% No-control RMS
figure; clf; hold all;
plotESP_Ahmed(std(data_contour_reset, 0, 2), Nx, Ny, resX, resY);
xlabel("Z");
ylabel("Y");
colorbar();
title('No control RMS');

% Percentage change in RMS
diff_std = 100 * (std(data_contour_step, 0, 2) - std(data_contour_reset, 0, 2)) ./ std(data_contour_reset, 0, 2);

% Higher resolution for the difference plot
resX = 100;
resY = 100;

figure; clf; hold all;
plotESP_Ahmed(diff_std, Nx, Ny, resX, resY);
xlabel('$Z\,(\mathrm{mm})$', 'Interpreter', 'latex');
ylabel('$Y\,(\mathrm{mm})$', 'Interpreter', 'latex');

hcb = colorbar();
caxis([-25 25]);
tick_vals   = hcb.Ticks;
tick_labels = strcat('$', string(tick_vals), '\%$');
set(hcb, 'Ticks', tick_vals, ...
         'TickLabels', tick_labels, ...
         'TickLabelInterpreter', 'latex');

colormap('bluewhitered');

set(gca, 'FontSize', 16);
set(gca, 'OuterPosition', [0, 0, 1, 1]);           % Maximize figure usage
set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.045)); % Tight layout































