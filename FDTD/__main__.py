#to do:
#   -make sceme more efficient-> dont save every time step only the current
#   -place observers
#   -triangular obstacle PEC and impedance










import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

#parameters
c=1 #wave speed[m/s]
Z=0 #impedance[ohm]

#geometry
a=25 #width [m]
b=15 #height [m]
d=4  #thickness of PML [gridcells]

#source position
xs=6 #[m]
ys=2 #[m]

#observer points
coordinates=[(5,10),(10,2)] #([m],[m])



# constructing function for k1 (x) or k2(y)
kappax=np.linspace(0,a,100)
kappay=10*(((kappax-(a-d))/d)**5 *np.heaviside(kappax-(a-d),1)-((kappax-d)/d)**5*np.heaviside(-kappax+d,1))
plt.plot(kappax,kappay)
plt.show()


#source in function of time:
Ps=lambda t:100*np.exp(-(t-3)**2)
plt.plot(np.linspace(0,20,282),Ps(np.linspace(0,20,282)))
plt.show()

#spatial steps
dx=0.1 #[m]
dy=0.1 #[m]

#time step based on CN and stabilaty
dt=np.sqrt(2)*0.05   #[s]
T=40 #time of simulation [s]

#cournat number
CN=c*dt/np.sqrt(dx**2+dy**2)  #[s/m]
print(CN)

#size of array
K=int(T/dt) #dimensioneless
N=int(a/dx) #dimensioneless
M=int(b/dy) #dimensioneless

print(f"M={M}")
print(f"N={N}")
print(f"K={K}")




#constructing mesh
ny, nx = M,N
x = np.arange(0, nx) * dx  #(N,1) matrix
y = np.arange(0, ny) * dy  #(M,1) matrix
X, Y = np.meshgrid(x, y)   #(M,N) matrix both

#kappa for x-direction
F1=10*(((X-(a-d))/d)**5 *np.heaviside(X-(a-d),1)-((X-d)/d)**5*np.heaviside(-X+d,1)) #(M,N) matrix
#plotting
plt.pcolormesh(X, Y, F1, shading='auto', cmap='viridis')
plt.colorbar(label='f(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('f(x, y) with pcolormesh')
plt.show()

#kappa for y-direction with floor
F2=10*(((Y-(b-d))/d)**5 *np.heaviside(Y-(b-d),1))
#plotting
plt.pcolormesh(X, Y, F2, shading='auto', cmap='viridis')
plt.colorbar(label='f(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('f(x, y) with pcolormesh')
plt.show()

#kappa for y-direction without floor
F3=10*(((Y-(b-d))/d)**5 *np.heaviside(Y-(b-d),1)-((Y-d)/d)**5*np.heaviside(-Y+d,1))

#plotting
plt.pcolormesh(X, Y, F3, shading='auto', cmap='viridis')
plt.colorbar(label='f(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('f(x, y) with pcolormesh')
plt.show()











#initialise
px=np.zeros((K,M,N))  # (M,N) matrix
py=np.zeros((K,M,N))  # (M,N) matrix
ox=np.zeros((K,M,N+1)) #(M,N+1) matrix
oy=np.zeros((K,M+1,N)) #(M+1,N) matrix
print(K)

#simulation in empty universe
def empty(px,py,ox,oy,F1,F3,K,xs,ys,co):


    #fix dimensions/construct the final kappa's (sligth alteration to dimension so they would fit in the equations)
    k1_o=F1[:,:-1]
    k1_p=F1
    k2_o=F3[1:,:]
    k2_p=F3

    #source index
    xi=int(ys/a*N)
    yi=int(xs/b*M)

    #observation
    observations=[]
    for i in range(len(co)):
        observations.append([])


    #solving scheme
    for i in range(K-1):

        #construct total p
        p=px[i]+py[i]
        #two equations for o
        ox[i+1,:,1:-1]=((1-k1_o*dt/2)/(1+k1_o*dt/2))*ox[i,:,1:-1]-dt/(dx*(1+k1_o*dt/2))*(p[:,1:]-p[:,:-1])
        oy[i+1,1:-1,:]=((1-k2_o*dt/2)/(1+k2_o*dt/2))*oy[i,1:-1,:]-dt/(dy*(1+k2_o*dt/2))*(p[1:,:]-p[:-1,:])

        
        

        #two equations for p
        px[i+1]=((1-k1_p*dt/2)/(1+k1_p*dt/2))*px[i]-(c**2/(1+k1_p*dt/2))*(dt/dx)*(ox[i+1,:,1:]-ox[i+1,:,:-1])
        py[i+1]=((1-k2_p*dt/2)/(1+k2_p*dt/2))*py[i]-(c**2/(1+k2_p*dt/2))*(dt/dy)*(oy[i+1,1:,:]-oy[i+1,:-1,:])

        

        # adding source
        px[i+1,xi,yi]+=Ps(i*dt)/2
        py[i+1,xi,yi]+=Ps(i*dt)/2
        #observing:
        for ind,j in enumerate(co):
            observations[ind].append(px[i+1,int(j[1]/b*M),int(j[0]/a*N)]+py[i+1,int(j[1]/b*M),int(j[0]/a*N)])
    return px+py,ox,oy

#simulation with floor and wall
def wall(px,py,ox,oy,F1,F2,K,xs,ys,co):


    #fix dimensions/construct the final kappa's (sligth alteration to dimension so they would fit in the equations)
    k1_o=F1[:,:-1] 
    k1_p=F1
    k2_o=F2[1:,:]
    k2_p=F2
    
    #source index
    xi=int(ys/a*N)
    yi=int(xs/b*M)
    
    #observation
    observations=[]
    for i in range(len(co)):
        observations.append([])



    #solving scheme
    for i in range(K-1):

        wall_left=int(8/25*(N+1))
        ceiling=int(7/b*(M+1))




        #construct total p
        p=px[i]+py[i]
        #two equations for o
        ox[i+1,:,1:-1]=((1-k1_o*dt/2)/(1+k1_o*dt/2))*ox[i,:,1:-1]-dt/(dx*(1+k1_o*dt/2))*(p[:,1:]-p[:,:-1])
        oy[i+1,1:-1,:]=((1-k2_o*dt/2)/(1+k2_o*dt/2))*oy[i,1:-1,:]-dt/(dy*(1+k2_o*dt/2))*(p[1:,:]-p[:-1,:])

        #aplying boundry conditions
        ox[i+1,:ceiling,wall_left]=0
        

        #two equations for p
        px[i+1]=((1-k1_p*dt/2)/(1+k1_p*dt/2))*px[i]-(c**2/(1+k1_p*dt/2))*(dt/dx)*(ox[i+1,:,1:]-ox[i+1,:,:-1])
        py[i+1]=((1-k2_p*dt/2)/(1+k2_p*dt/2))*py[i]-(c**2/(1+k2_p*dt/2))*(dt/dy)*(oy[i+1,1:,:]-oy[i+1,:-1,:])

        

        # adding source
        px[i+1,xi,yi]+=Ps(i*dt)/2
        py[i+1,xi,yi]+=Ps(i*dt)/2

        #observing:
        for ind,j in enumerate(co):
            observations[ind].append(px[i+1,int(j[1]/b*M),int(j[0]/a*N)]+py[i+1,int(j[1]/b*M),int(j[0]/a*N)])
    return px+py,ox,oy,observations

#simulation with floor and rectangle
def rectangle(px,py,ox,oy,F1,F2,K,Z,xs,ys,co):
    #fix dimensions/construct the final kappa's (sligth alteration to dimension so they would fit in the equations)
    k1_o=F1[:,:-1]
    k1_p=F1
    k2_o=F2[1:,:]
    k2_p=F2

    #source index
    xi=int(ys/a*N)
    yi=int(xs/b*M)


    #observation
    observations=[]
    for i in range(len(co)):
        observations.append([])

    
    #determie obstacle indexes
    wall_left=int(8/25*(N+1))
    wall_right=int(12/25*(N+1))
    ceiling=int(7/b*(M+1))

    #solving scheme
    for i in range(K-1):
        #construct total p
        p=px[i]+py[i]
        #two equations for o
        ox[i+1,:,1:-1]=((1-k1_o*dt/2)/(1+k1_o*dt/2))*ox[i,:,1:-1]-dt/(dx*(1+k1_o*dt/2))*(p[:,1:]-p[:,:-1])
        oy[i+1,1:-1,:]=((1-k2_o*dt/2)/(1+k2_o*dt/2))*oy[i,1:-1,:]-dt/(dy*(1+k2_o*dt/2))*(p[1:,:]-p[:-1,:])

        #aplying boundry conditions
        ox[i+1,:ceiling,wall_left]=((1-dt*Z/dx)*ox[i,:ceiling,wall_left]+2*dt/dx*p[:ceiling,wall_left-1])/(1+Z*dt/dx)
        ox[i+1,:ceiling,wall_right]=((1-dt*Z/dx)*ox[i,:ceiling,wall_right]-2*dt/dx*p[:ceiling,wall_right])/(1+Z*dt/dx)
        oy[i+1,ceiling,wall_left:wall_right]=((1-dt*Z/dx)*oy[i,ceiling,wall_left:wall_right]-2*dt/dx*p[ceiling,wall_left:wall_right])/(1+Z*dt/dx)

        #two equations for p
        px[i+1]=((1-k1_p*dt/2)/(1+k1_p*dt/2))*px[i]-(c**2/(1+k1_p*dt/2))*(dt/dx)*(ox[i+1,:,1:]-ox[i+1,:,:-1])
        py[i+1]=((1-k2_p*dt/2)/(1+k2_p*dt/2))*py[i]-(c**2/(1+k2_p*dt/2))*(dt/dy)*(oy[i+1,1:,:]-oy[i+1,:-1,:])

        #removing inside obstacle ->not sure if neccecary
        px[i+1,:ceiling,wall_left:wall_right]=0
        py[i+1,:ceiling,wall_left:wall_right]=0

        # adding source
        px[i+1,xi,yi]+=Ps(i*dt)/2
        py[i+1,xi,yi]+=Ps(i*dt)/2

        #observing:
        for ind,j in enumerate(co):
            observations[ind].append(px[i+1,int(j[1]/b*M),int(j[0]/a*N)]+py[i+1,int(j[1]/b*M),int(j[0]/a*N)])
    return px+py,ox,oy
        
# simulation with triangle
def triangle(px,py,ox,oy,F1,F2,K,xs,ys,co):
    #fix dimensions/construct the final kappa's (sligth alteration to dimension so they would fit in the equations)
    k1_o=F1[:,:-1]
    k1_p=F1
    k2_o=F2[1:,:]
    k2_p=F2

    #source index
    xi=int(ys/a*N)
    yi=int(xs/b*M)

    #observation
    observations=[]
    for i in range(len(co)):
        observations.append([])

    # returns True if x and y lie in the triangle
    def in_triangle(x, y):
        left_x = 6 + 2 # leftmost x of triangle [m]
        right_x = 6 + 2 + 2 # rightmost         [m]
        upper_y = 2 * 2 # uppermost y           [m]
        # to be in triangle is the same as being beneath two lines
        return (y < upper_y*(x - left_x)) & (y < -upper_y * (x - right_x))
    
    # discretisation points
    x_ox = np.linspace(0, a, N+1).reshape((1,   N+1))
    y_ox = np.linspace(0, b, M)  .reshape((M,   1))
    x_oy = np.linspace(0, a, N)  .reshape((1,   N))
    y_oy = np.linspace(0, b, M+1).reshape((M+1, 1))

    TRIANGLE_ox = in_triangle(x_ox, y_ox)
    TRIANGLE_oy = in_triangle(x_oy, y_oy)

    #solving scheme
    for i in range(K-1):
        #construct total p
        p=px[i]+py[i]
        #two equations for o
        ox[i+1,:,1:-1]=((1-k1_o*dt/2)/(1+k1_o*dt/2))*ox[i,:,1:-1]-dt/(dx*(1+k1_o*dt/2))*(p[:,1:]-p[:,:-1])
        oy[i+1,1:-1,:]=((1-k2_o*dt/2)/(1+k2_o*dt/2))*oy[i,1:-1,:]-dt/(dy*(1+k2_o*dt/2))*(p[1:,:]-p[:-1,:])

        #aplying boundry conditions
        ox[i+1,TRIANGLE_ox]=0
        oy[i+1,TRIANGLE_oy]=0

        #two equations for p
        px[i+1]=((1-k1_p*dt/2)/(1+k1_p*dt/2))*px[i]-(c**2/(1+k1_p*dt/2))*(dt/dx)*(ox[i+1,:,1:]-ox[i+1,:,:-1])
        py[i+1]=((1-k2_p*dt/2)/(1+k2_p*dt/2))*py[i]-(c**2/(1+k2_p*dt/2))*(dt/dy)*(oy[i+1,1:,:]-oy[i+1,:-1,:])

        # adding source
        px[i+1,xi,yi]+=Ps(i*dt)/2
        py[i+1,xi,yi]+=Ps(i*dt)/2

        #observing:
        for ind,j in enumerate(co):
            observations[ind].append(px[i+1,int(j[1]/b*M),int(j[0]/a*N)]+py[i+1,int(j[1]/b*M),int(j[0]/a*N)])
    return px+py,ox,oy,observations




p,ox,oy,obs=triangle(px,py,ox,oy,F1,F3,K,xs,ys,coordinates)



#animate function
def animate_heatmap(frames, interval=100, cmap='viridis', save_path=None, fps=10):
    """
    Animate a sequence of 2D numpy arrays (frames) as a heatmap.
    
    Parameters
    ----------
    frames : np.ndarray
        3D array of shape (num_frames, height, width)
    interval : int, optional
        Time between frames in milliseconds (default 100)
    cmap : str, optional
        Colormap for the heatmap (default 'viridis')
    save_path : str, optional
        If provided, saves animation to this path (e.g., 'animation.gif')
    fps : int, optional
        Frames per second for saved animation (default 10)
    """
    # Sanity check
    if frames.ndim != 3:
        raise ValueError("Input 'frames' must be a 3D NumPy array (num_frames, height, width).")
    
    num_frames, height, width = frames.shape

    # Create figure
    fig, ax = plt.subplots()
    heatmap = ax.imshow(frames[0], cmap=cmap, interpolation='nearest',origin='lower')   #,vmin=frames.min(), vmax=frames.max()/4)
    plt.colorbar(heatmap, ax=ax)

    # Update function
    def update(frame_idx):
        heatmap.set_data(frames[frame_idx])
        ax.set_title(f"Frame {frame_idx}/{num_frames}")
        return [heatmap]

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                  interval=interval, blit=True)

    # Save if requested
    if save_path:
        if save_path.endswith('.gif'):
            ani.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith('.mp4'):
            ani.save(save_path, writer='ffmpeg', fps=fps)
        else:
            raise ValueError("save_path must end with '.gif' or '.mp4'")
    else:
        plt.show()

    return ani

animate_heatmap(p)

t=np.linspace(0,T,int(T/dt)-1)

plt.plot(t,obs[1])
plt.show()