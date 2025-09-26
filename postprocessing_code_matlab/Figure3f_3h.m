V_u = load('/Volumes/Untitled/REACT_data/PIV_data/UC/DataMatrixU.mat').DataMatrixU;
V_v = load('/Volumes/Untitled/REACT_data/PIV_data/UC/DataMatrixW.mat').DataMatrixV;

Nx = 215;
Ny = 106;

grid_shape = [Ny, Nx];

num_steps = size(V_u, 2);

% Pre-allocate

datamatrix_uc_up = zeros(Nx * Ny, num_steps);
datamatrix__uc_vp = zeros(Nx * Ny, num_steps);


dx = 1;
dy = 1;

% First compute mean fields
U_mean = mean(V_u, 2);  % (Nx*Ny, 1)
V_mean = mean(V_v, 2);


for step = 1:num_steps
    U = reshape(V_u(:, step), grid_shape);
    V = reshape(V_v(:, step), grid_shape);
    
    % Velocity fluctuations
    u_prime = V_u(:, step) - U_mean;
    v_prime = V_v(:, step) - V_mean;
    


    % Store in data matrices
    datamatrix_uc_up(:, step) = u_prime(:);
    datamatrix_uc_vp(:, step) = v_prime(:);

    disp(['Step ' num2str(step) ' complete']);
end


%% 
V_u = load('/Volumes/Untitled/REACT_data/PIV_data/RL_C/DataMatrixU.mat').DataMatrixU;
V_v = load('/Volumes/Untitled/REACT_data/PIV_data/RL_C/DataMatrixW.mat').DataMatrixV;

Nx = 215;
Ny = 106;

grid_shape = [Ny, Nx];

num_steps = size(V_u, 2);

% Pre-allocate

datamatrix_rlc_up = zeros(Nx * Ny, num_steps);
datamatrix_rlc_vp = zeros(Nx * Ny, num_steps);


dx = 1;
dy = 1;

% First compute mean fields
U_mean = mean(V_u, 2);  % (Nx*Ny, 1)
V_mean = mean(V_v, 2);


for step = 1:num_steps
    U = reshape(V_u(:, step), grid_shape);
    V = reshape(V_v(:, step), grid_shape);
    
    % Velocity fluctuations
    u_prime = V_u(:, step) - U_mean;
    v_prime = V_v(:, step) - V_mean;
    


    % Store in data matrices
    datamatrix_rlc_up(:, step) = u_prime(:);
    datamatrix_rlc_vp(:, step) = v_prime(:);

    disp(['Step ' num2str(step) ' complete']);
end

%% Integrated POD Computation
[U_cf, S_cf, V_cf] = pod([datamatrix_rlc_up; datamatrix_rlc_vp]');
[U_baseline, S_baseline, V_baseline] = pod([datamatrix_uc_up;datamatrix_uc_vp]');




%% Total PSD secondversion
nwindow = 2^12;
FsRT = 100;
fscale      = 0.216/15;

p_base = zeros(2049, 1);
p_cf = zeros(2049, 1);
p_sn = zeros(2049, 1);
step = 0;
selection_range = 50;
for i = 1:selection_range
    step = step + 1
    [p_basetmp, f_basetmp] = pwelch(detrend(S_baseline(i)*V_baseline(:, i)),hanning(nwindow),0.5*nwindow,nwindow,FsRT*fscale);
    [p_cftmp, f_cftmp] = pwelch(detrend(S_cf(i)*V_cf(:, i)),hanning(nwindow),0.5*nwindow,nwindow,FsRT*fscale);
   
    p_base = p_base + p_basetmp;
    p_cf = p_cf +  p_cftmp;
    
    f_base = f_basetmp;
    f_cf = f_cftmp;
  


end

p_base = p_base/selection_range;
p_cf = p_cf/selection_range;

smoothed_base = smooth(f_base .* p_base, 30);  
smoothed_cf = smooth(f_cf .* p_cf, 30);


figure;
set(gcf, 'Position', [100, 100, 800, 445.93]);


set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.02));
set(gcf, 'PaperPositionMode', 'auto');


color_base = [0.3, 0.3, 0.3] 
color_cf = [0.6 0.4 0.8];   

semilogx(f_base, smoothed_base, '-', 'Color', color_base, 'LineWidth', 3);
hold on;
semilogx(f_cf, smoothed_cf, '-', 'Color', color_cf, 'LineWidth', 3);


xl = xlabel('$St$', 'Interpreter', 'latex', 'FontSize', 25);
yl = ylabel('$St \cdot \Phi_{u,w}(St)$', 'Interpreter', 'latex', 'FontSize', 25);
xlim([3.5e-4, 0.61])

legend('Uncontrolled (UC)', 'RL controlled (RL-C)', 'Location', 'northwest', 'FontSize', 20)
set(gca, 'FontSize', 26);  
set(gca, 'LineWidth', 2);   

%% %% 
[U_up_baseline, S_up_baseline, V_up_baseline] = pod(datamatrix_uc_up');
[U_up_cf, S_up_cf, V_up_cf] = pod(datamatrix_rlc_up');



%% POD Energy as Bar Chart (First 10 Modes Only, Mode-wise Comparison)
figure('Position', [100, 100, 675, 250]);  
set(gcf, 'Color', 'w');  

num_modes = 10;
x_values = 1:num_modes;

S_baseline_trimmed = S_up_baseline(1:num_modes).^2;
S_cf_trimmed = S_up_cf(1:num_modes).^2;

S_cf_trimmed_4 = S_cf_trimmed(3);
S_cf_trimmed(3) = S_cf_trimmed(4);
S_cf_trimmed(4) = S_cf_trimmed_4;

% Data for bar chart
data_matrix = [S_baseline_trimmed, S_cf_trimmed];

color_base = [0.3 0.3 0.3];      
color_cf = [0.6 0.4 0.8];   

% Bar chart with mode-wise comparison
b = bar(x_values, data_matrix, 'grouped', 'BarWidth', 0.8);
b(1).FaceColor = color_base
b(1).FaceAlpha = 0.41;
b(1).EdgeColor = 'k';
b(1).LineWidth = 1.2;

b(2).FaceColor = color_cf
b(2).FaceAlpha = 0.41;
b(2).EdgeColor = 'k';
b(2).LineWidth = 1.2;

% Add legend with improved styling
legend({'Uncontrolled', 'RL Controlled'}, 'FontSize', 12, 'Location', 'northeast', 'Box', 'off');

% Label settings
xlabel('Modes', 'FontSize', 16, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
ylabel('Modal Energy', 'FontSize', 16, 'FontName', 'Times New Roman', 'Interpreter', 'latex');


% Grid and axes settings
ax = gca;
ax.FontSize = 14;
ax.LineWidth = 1.2;
ax.GridAlpha = 0.3;

%% PSD of mode 

nwindow = 2^8;
FsRT = 100;
mode = 2
fscale      = 0.216/15;

mode = 1; %Note that mode 3 and mode 4 switch position from control to uncontrol
[pxxQ22, f_PSDQ22] = pwelch(S_up_baseline(mode) * V_up_baseline(:, mode),hanning(nwindow),0.5*nwindow,nwindow,FsRT*fscale);
[pxxQ23, f_PSDQ23] = pwelch(S_up_cf(mode) * V_up_cf(:, mode),hanning(nwindow),0.5*nwindow,nwindow,FsRT*fscale);


color_base = [0.3 0.3 0.3];      
color_cf = [0.6 0.4 0.8];  

figure('Position', [100, 100, 490, 680], 'Color', 'w');
semilogx(f_PSDQ22, f_PSDQ22 .* pxxQ22, 'LineWidth', 4, 'Color', color_base); hold on;
semilogx(f_PSDQ23, f_PSDQ23 .* pxxQ23, 'LineWidth', 4, 'Color', color_cf);


xl = xlabel('$St$', 'Interpreter', 'latex', 'FontSize', 24);
yl = ylabel('$St \cdot \Phi(St)$', 'Interpreter', 'latex', 'FontSize', 24);
set(gca, 'FontSize', 32); 
set(gca, 'LineWidth', 2);    


%% Group Plot export

grid_shape = [106, 215];

x = linspace(0, 1, grid_shape(2));
y = linspace(0, 1, grid_shape(1));
[X_grid, Y_grid] = meshgrid(x, y);
[Xg, Yg] = meshgrid(x, y);

X_rotated = Y_grid';  
Y_rotated = flipud(X_grid');  

nRows = 2;
nCols = 3;

sigma = 1;
mode_indices = [1, 3, 4, 1, 4, 3];  % Example modes


figure('Position', [100, 100, 1250, 500], 'Color', 'w');  % Adjust overall figure size to fit all subplots

for i = 1:6
    
    if i < 4
        S_tmp = S_baseline;
        U_tmp = U_baseline;
    else
        S_tmp = S_cf;
        U_tmp = U_cf;
    end
    

    subplot('Position', [(mod(i-1, 3) * 0.3) + 0.07, 0.61 - (floor((i-1) / 3) * 0.46), 0.25, 0.36]);
    
    mode_spatial = real(S_tmp(mode_indices(i)) * U_tmp(mode_indices(i), :));
    mode_reshaped = reshape(mode_spatial, grid_shape);

    % Apply Gaussian filter
    mode_smoothed = imgaussfilt(mode_reshaped, sigma);
    mode_rotated = rot90(mode_smoothed, 2);  % 90° counter-clockwise
    if i ==1
        mode_rotated = -mode_rotated
    end

    % Plot contourf
    contourf(X_grid, Y_grid, -mode_rotated, 100, 'LineColor', 'none');
    if i == 4 || i == 1  % Corrected logical condition
        caxis([-200, 200]);  % Adjust color limits
    elseif i == 3
        caxis([-70, 80]);
    else
        caxis([-80, 80]);
    end
    xlabel('$z/W$', 'Interpreter', 'latex', 'FontSize', 14); 
    ylabel('$x/W$', 'Interpreter', 'latex', 'FontSize', 14); 
    xticks(0:0.5:1)  
    ylim([0, 1]);                          
    yticks([0, 0.5, 1]);                  

    yticklabels({'0.6', '0.3', '0'});      % Custom labels
    set(gca, 'FontSize', 18);  % Tick label font size

    set(gca, 'YDir', 'normal');
    cmocean('thermal');
    axis off

end












































