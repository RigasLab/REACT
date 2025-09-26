%Preparing Reynolds stress and turbulent production
V_u = load('/Volumes/Untitled/REACT_data/PIV_data/UC/DataMatrixU.mat').DataMatrixU;
V_v = load('/Volumes/Untitled/REACT_data/PIV_data/UC/DataMatrixW.mat').DataMatrixV;

Nx = 215;
Ny = 106;

grid_shape = [Ny, Nx];

num_steps = size(V_u, 2);

% Pre-allocate

datamatrix_base_uu = zeros(Nx * Ny, num_steps);
datamatrix_base_vv = zeros(Nx * Ny, num_steps);
datamatrix_base_uv = zeros(Nx * Ny, num_steps);


dx = 1;
dy = 1;

% First compute mean fields
U_mean = mean(V_u, 2);  
V_mean = mean(V_v, 2);

U_mean = reshape(U_mean, grid_shape);
[dUdx, dUdy] = gradient(U_mean, dx, dy);  

U_mean = reshape(U_mean, [Nx*Ny, 1]);


for step = 1:num_steps
    U = reshape(V_u(:, step), grid_shape);
    V = reshape(V_v(:, step), grid_shape);
    
    % Velocity fluctuations
    u_prime = reshape(V_u(:, step) - U_mean, grid_shape);
    v_prime = reshape(V_v(:, step) - V_mean, grid_shape);
    
    % Reynolds stress terms
    uu = u_prime .* u_prime;
    vv = v_prime .* v_prime;
    uv = u_prime .* v_prime;
    P = uv .* dUdx;

    % Store in data matrices
    datamatrix_base_uu(:, step) = uu(:);
    datamatrix_base_vv(:, step) = vv(:);
    datamatrix_base_uv(:, step) = uv(:);


    disp(['Step ' num2str(step) ' complete']);
end


V_u = load('/Volumes/Untitled/REACT_data/PIV_data/RL_C/DataMatrixU.mat').DataMatrixU;
V_v = load('/Volumes/Untitled/REACT_data/PIV_data/RL_C/DataMatrixW.mat').DataMatrixV;

Nx = 215;
Ny = 106;

grid_shape = [Ny, Nx];

num_steps = size(V_u, 2);

% Pre-allocate

datamatrix_cf_uu = zeros(Nx * Ny, num_steps);
datamatrix_cf_vv = zeros(Nx * Ny, num_steps);
datamatrix_cf_uv = zeros(Nx * Ny, num_steps);


dx = 1;
dy = 1;

% First compute mean fields
U_mean = mean(V_u, 2);  
V_mean = mean(V_v, 2);

U_mean = reshape(U_mean, grid_shape);
[dUdx, dUdy] = gradient(U_mean, dx, dy);  

U_mean = reshape(U_mean, [Nx*Ny, 1]);



for step = 1:num_steps
    U = reshape(V_u(:, step), grid_shape);
    V = reshape(V_v(:, step), grid_shape);
    
    % Velocity fluctuations
    u_prime = reshape(V_u(:, step) - U_mean, grid_shape);
    v_prime = reshape(V_v(:, step) - V_mean, grid_shape);
    
    % Reynolds stress terms
    uu = u_prime .* u_prime;
    vv = v_prime .* v_prime;
    uv = u_prime .* v_prime;
    P = uv .* dUdx;

    % Store in data matrices
    datamatrix_cf_uu(:, step) = uu(:);
    datamatrix_cf_vv(:, step) = vv(:);
    datamatrix_cf_uv(:, step) = uv(:);


    disp(['Step ' num2str(step) ' complete']);
end


mean_cfuu = mean(datamatrix_cf_uu, 2);  % u'u'
mean_cfvv = mean(datamatrix_cf_vv, 2);  % v'v'
mean_cfuv = mean(datamatrix_cf_uv, 2);  % u'v'


mean_baseuu = mean(datamatrix_base_uu, 2);  % u'u'
mean_basevv = mean(datamatrix_base_vv, 2);  % v'v'
mean_baseuv = mean(datamatrix_base_uv, 2);  % u'v'



Nx = 215;
Ny = 106;
grid_shape = [Ny, Nx];
x = linspace(0, 1, grid_shape(2));
y = linspace(0, 1, grid_shape(1));
[X_grid, Y_grid] = meshgrid(x, y);

V_u = load('/Volumes/Untitled/REACT_data/PIV_data/RL_C/DataMatrixU.mat').DataMatrixU;
dx = 1;
dy =1;
U_mean = mean(V_u, 2);  % (Nx*Ny, 1)
U_mean = reshape(U_mean, grid_shape);
[dUdx_cf, dUdy_cf] = gradient(U_mean, dx, dy);  
P_cf =  reshape(mean_cfuv, grid_shape) .* dUdx_cf;  

V_u = load('/Volumes/Untitled/REACT_data/PIV_data/UC/DataMatrixU.mat').DataMatrixU;
dx = 1;
dy =1;
U_mean = mean(V_u, 2);  % (Nx*Ny, 1)
U_mean = reshape(U_mean, grid_shape);
[dUdx_base, dUdy_base] = gradient(U_mean, dx, dy);  
P_base =  reshape(mean_baseuv, grid_shape) .* dUdx_base;  



%% Plot reynolds stress and turbulent production contour

for i = 1:2
    % Select data and case title
    switch i
        case 1
            mean_uu = reshape(mean_cfuu, grid_shape);
            mean_vv = reshape(mean_cfvv, grid_shape);
            mean_uv = reshape(mean_cfuv, grid_shape);
            case_title = 'RL Controlled, ';
            P = P_cf;
        case 2
            mean_uu = reshape(mean_baseuu, grid_shape);
            mean_vv = reshape(mean_basevv, grid_shape);
            mean_uv = reshape(mean_baseuv, grid_shape);
            case_title = 'Uncontrolled, ';
            P = P_base;
        
    end

    fig = figure('Position', [100, 100, 1200, 260]);  % [left, bottom, width, height]
    t = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
    

    X_rotated = Y_grid';           % Swap X and Y
    Y_rotated = flipud(X_grid');   % Flip X and transpose


    nexttile;
    mode_rotate = rot90(P, 2);
    mode_smoothed = imgaussfilt(mode_rotate, 1);  % Optional smoothing
    contourf(X_grid, Y_grid, mode_rotate, 900, 'LineColor', 'none');
    colorbar;

    caxis([0, 2.6]);  % symmetric
    cmocean('thermal')
    title([case_title, '$\langle u'' v'' \rangle \cdot \frac{\partial \overline{U}}{\partial y}$'], ...
          'Interpreter', 'latex', 'FontSize', 16);
    xlabel('$z/W$', 'Interpreter', 'latex', 'FontSize', 14); 
    ylabel('$x/W$', 'Interpreter', 'latex', 'FontSize', 14); 
    xticks(0:0.5:1)  
        ylim([0, 1]);                          
    yticks([0, 0.5, 1]);                   
    yticklabels({'0.6', '0.3', '0'});      
    set(gca, 'FontSize', 16);  



    nexttile;
    mode_rotate = rot90(mean_uu, 2);
    contourf(X_grid, Y_grid, mode_rotate, 100, 'LineColor', 'none');

    colorbar;
    caxis([0, 10]);
    cmocean('thermal')
    title([case_title,'$\langle u'' u'' \rangle$'], 'Interpreter', 'latex', 'FontSize', 16);
 
    xlabel('$z/W$', 'Interpreter', 'latex', 'FontSize', 14); 
    ylabel('$x/W$', 'Interpreter', 'latex', 'FontSize', 14); 
    xticks(0:0.5:1)  % or use xticks([-1.02, -0.5, 0, 0.5, 1.02]) for explicit values
    ylim([0, 1]);                          % Real data extent
    yticks([0, 0.5, 1]);                   % Tick positions
    yticklabels({'0.6', '0.3', '0'});      % Custom labels
    set(gca, 'FontSize', 16);  % Tick label font size

    
    % v'v'
    nexttile;
    mode_rotate = rot90(mean_vv, 2);
    contourf(X_grid, Y_grid, mode_rotate, 100, 'LineColor', 'none');
    ylim([0.05,1])
    colorbar;
    caxis([0, 6]);
    cmocean('thermal')
    title([case_title,'$\langle v'' v'' \rangle$'], 'Interpreter', 'latex', 'FontSize', 16);


    xlabel('$z/W$', 'Interpreter', 'latex', 'FontSize', 14); 
    ylabel('$x/W$', 'Interpreter', 'latex', 'FontSize', 14); 
    xticks(0:0.5:1)  % or use xticks([-1.02, -0.5, 0, 0.5, 1.02]) for explicit values
        ylim([0, 1]);                          % Real data extent
    yticks([0, 0.5, 1]);                   % Tick positions
    yticklabels({'0.6', '0.3', '0'});      % Custom labels
    set(gca, 'FontSize', 16);  % Tick label font size

end

%% Spanwise integral


P_cf_integral = sum(P_cf(:, 4:213), 2); %Trim the flap region [4:213]
P_base_integral = sum(P_base(:, 4:213), 2);

mean_cfuu = reshape(mean_cfuu, grid_shape);
mean_baseuu = reshape(mean_baseuu, grid_shape);

mean_cfvv = reshape(mean_cfvv, grid_shape);
mean_basevv = reshape(mean_basevv, grid_shape);

mean_cfuv = reshape(mean_cfuv, grid_shape);
mean_baseuv = reshape(mean_baseuv, grid_shape);

mean_cfuu_integral = sum(mean_cfuu(:, 4:213), 2);
mean_baseuu_integral = sum(mean_baseuu(:, 4:213), 2);

mean_cfvv_integral = sum(mean_cfvv, 2);
mean_basevv_integral = sum(mean_basevv, 2);

mean_cfuv_integral = sum(mean_cfuv(:, 4:213), 2);
mean_baseuv_integral = sum(mean_baseuv(:, 4:213), 2);

dudx_cf_integral = sum(dUdx_cf(:, 4:213), 2);
dudx_base_integral = sum(dUdx_base(:, 4:213), 2);
%% Spanwise integral curve

set(0, 'DefaultLineLineWidth', 3);
set(0, 'DefaultAxesFontSize', 14);
set(0, 'DefaultAxesTickLabelInterpreter', 'latex');
set(0, 'DefaultLegendInterpreter', 'latex');
set(0, 'DefaultTextInterpreter', 'latex');

% Define custom colors
color_base = [0.3 0.3 0.3];             % Black
color_snaking = [0.2 0.7 0.2];    % Greenish
color_cf = [0.6 0.4 0.8];         % Purple-ish

% --- First Plot ---
figure('Position', [100, 100, 540, 567]);
hold on
plot(P_base_integral, 'Color', color_base, 'LineStyle', '-', 'LineWidth', 4, 'DisplayName', 'Uncontrolled');
plot(P_cf_integral, 'Color', color_cf, 'LineWidth', 4, 'DisplayName', 'RL Controlled');
xlim([5, 100])
ylim([0, 120])
box on

legend('Location', 'southeast','FontSize', 18)

xticks([0, 25, 50,75, 100]);                   % Tick positions
xticklabels({'0', '0.15', '0.3','0.45', '0.6'});      % Custom labels
yticks([0 40 80 120]);

set(gca, 'FontSize', 30);  % Tick label font size
set(gca, 'LineWidth', 2);     % Increase axis line thickness

xlabel('$x/W$', 'Interpreter', 'latex', 'FontSize', 30)
ylabel('$\int \langle u'' w'' \rangle \frac{\partial \langle u \rangle}{\partial z} \, dz$', 'FontSize', 32)

% --- Second Plot ---
figure('Position', [100, 100, 540, 567]);
hold on
plot(mean_baseuu_integral, 'Color', color_base, 'LineStyle', '-', 'LineWidth', 4,'DisplayName', 'Uncontrolled');
plot(mean_cfuu_integral, 'Color', color_cf, 'LineWidth', 4, 'DisplayName', 'RL Controlled');
xlim([3, 100])
ylim([400, 1200])

box on


xticks([0, 25, 50,75, 100]);                   % Tick positions
xticklabels({'0', '0.15', '0.3','0.45', '0.6'});      % Custom labels
yticks([400 600 800 1000 1200]);
set(gca, 'FontSize', 30);  % Tick label font size
set(gca, 'LineWidth', 2);     % Increase axis line thickness

ylabel('$\int \langle u'' u'' \rangle \, dz$', 'FontSize', 30)
legend('Location', 'southeast', 'FontSize', 18)
xlabel('$x/W$', 'Interpreter', 'latex', 'FontSize', 32)


% --- Third Plot ---
figure('Position', [100, 100, 540, 567]);
hold on
plot(mean_basevv_integral, 'Color', color_base, 'LineStyle', '-', 'LineWidth', 4, 'DisplayName', 'Uncontrolled');
plot(mean_cfvv_integral, 'Color', color_cf, 'LineWidth', 4, 'DisplayName', 'RL Controlled');
xlim([3, 100])
box on

xticks([0, 25, 50,75, 100]);                   % Tick positions
xticklabels({'0', '0.15', '0.3','0.45', '0.6'});      % Custom labels
yticks([200 400 600 800]);
set(gca, 'FontSize', 30);  % Tick label font size
set(gca, 'LineWidth', 2);     % Increase axis line thickness
ylabel('$\int \langle w'' w'' \rangle \, dz$', 'FontSize', 30)
legend('Location', 'southeast','FontSize', 18)
xlabel('$x/W$', 'Interpreter', 'latex', 'FontSize', 32)




