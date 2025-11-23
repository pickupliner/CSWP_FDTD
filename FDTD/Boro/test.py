import numpy as np

dx=0.1 #[m]
dy=0.1 #[m]
M=100
N=101
ny, nx = M,N
x = np.arange(0, nx) * dx  
y = np.arange(0, ny) * dy
X, Y = np.meshgrid(x, y)  

xx = np.shape(x)
yy = np.shape(y)
meshx = np.shape(X)
meshy = np.shape(Y)
print(x)
print(xx,yy,meshx,meshy)
