function [] = plotESP_Ahmed(ESPdata, Nx, Ny, resX, resY)
%plots the ESP data in a square of sizes sizeX x sizeY with a resolution of
%resX x resY. The plot is centred with the square, with positive values
%going rightwards and upwards
%inputs ESPdata is the data in format 1xN, with N = Nx x Ny. The data in
%           ESPdata starts in the top left corner and increases to the
%           right in horizontal and to the bottom in vertical
%       Nx and Ny are the original horizontal and vertical resolution
%       resX and resY are the resolution of the plot

sizeX = 181.02; %mm
sizeY = 134.12; %mm

sizeBX = 216; %mm (size body X)
sizeBY = 160; %mm (size body Y)
BX = [-0.5,+0.5,+0.5,-0.5,-0.5] * sizeBX;
BY = [+0.5,+0.5,-0.5,-0.5,+0.5] * sizeBY;

%First, X and Y values are obtained:
%sample points
xs = ((1:1:Nx) - (Nx+1)/2)/((Nx-1)/2)*sizeX/2;
ys = -((1:1:Ny) - (Ny+1)/2)/((Ny-1)/2)*sizeY/2;
[Xs,Ys] = meshgrid(xs,ys);
%query points
xq = ((1:1:resX) - (resX+1)/2)/((resX-1)/2)*sizeX/2;
yq = -((1:1:resY) - (resY+1)/2)/((resY-1)/2)*sizeY/2;
[Xq,Yq] = meshgrid(xq,yq);

ESPs = reshape(ESPdata,[Nx, Ny])';

ESPq = interp2(Xs,Ys,ESPs,Xq,Yq,'spline');

%clf
plot(BX,BY,'k');
hold on;
pcolor(Xq,Yq,ESPq);
shading interp;
%contour(Xq,Yq,ESPq,8,'k', 'LineWidth', 0.05)
contourf(Xq, Yq, ESPq, 26, 'LineColor', 'none')

%plot(Xs(:),Ys(:),'ok');
plot(Xs(:),Ys(:),'.k');
%contourf(Xq,Yq,ESPq,16,'k')
axis equal
set(gca,'TickLabelInterpreter','latex');
xlim([min(BX),max(BX)]*1);
ylim([min(BY),max(BY)]*1);

end