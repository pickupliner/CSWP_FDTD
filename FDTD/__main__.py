#to do:
#   -make sceme more efficient-> dont save every time step only the current
#   -place observers
#   -triangular obstacle PEC and impedance










import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

#plotting booleans
PML=False
refinement=False
#parameters
c=1 #wave speed[m/s]
Z=2 #impedance[ohm]

#geometry
distance=5.6 #fundamental distance
a=7*distance  #width [m]
b=4*distance #height [m]
d=4  #thickness of PML [gridcells]
wl=2*distance # distance of wall
wr=3*distance #distance of seccond wall
wh= 2*distance                  #height of walls/ceiling
#source position
xs=distance #[m]
ys=distance/10 #[m]

#observer points
coordinates1=[[3*distance,distance/2],[4*distance,distance/2],[5*distance,distance/2]] #([m],[m]) for thin wall
coordinates2=[[4*distance,distance/2],[5*distance,distance/2],[6*distance,distance/2]] #([m],[m]) for other obstacles


# constructing function for k1 (x) or k2(y)
kappax=np.linspace(0,a,100)
kappay=10*(((kappax-(a-d))/d)**5 *np.heaviside(kappax-(a-d),1)-((kappax-d)/d)**5*np.heaviside(-kappax+d,1))
if PML:
    plt.plot(kappax,kappay)
    plt.show()


#source in function of time:
f=4.95*c/(distance*2*np.pi)
sigma=f
t0=18
Ps=lambda t:10*np.sin(2*np.pi*f*(t-t0))*np.exp(-((t-t0)**2)*(sigma**2))
#Ps = lambda t: 10*np.exp(-(t - .5)**2*16) # source
plt.plot(np.linspace(0,20,282),Ps(np.linspace(0,20,282)))
plt.show()

#size of array
K=int(665) #dimensioneless
N=int(250) #dimensioneless
M=int(150) #dimensioneless

print(f"M={M}")
print(f"N={N}")
print(f"K={K}")


def coordTransform(x, x0, x1, f):
    xM = np.max(x)
    # dx = f * dx'
    # xM = dx * (x1 - x0) + dx' * (xM - (x1 - x0))
    # <-> xM = dx' (xM + (f - 1) (x1 - x0))
    slope_rem = xM / (xM + (f - 1)*(x1 - x0))
    slope = f*slope_rem
    # # <-> xM - dx (x1 - x0) = dx' (xM - x1 + x0)
    # # <-> dx' = (xM - dx (x1 - x0)) / (xM - x1 + x0) 
    # slope_rem = (xM - slope*(x1 - x0)) / (xM - x1 + x0)
    u = np.where(x < x0, slope_rem*x, 
                 np.where(x < x1, slope*(x - x0) + slope_rem*x0, 
                          xM - slope_rem*(xM - x)))
    return u
def parabolicCoordTransform(x, x0, x1, f, buffer):
    xM = np.max(x)
    # if buffer on left hand side goes beneath x=0
    b_l = min(x0, buffer)
    b_r = min(buffer, xM - x1)
    slope_rem = xM / (xM + (f - 1)*(x1 - x0 + (b_l + b_r)/2))
    slope = f*slope_rem
    def first_lin(x):
        return slope_rem*x
    def left_para(x):
        return (x - x0 + buffer)**2 /2 * (slope - slope_rem)/buffer + slope_rem*x
    def second_lin(x):
        return slope*(x - x0) + max(left_para(x0), 0)
    def right_para(x):
        return (x - x1)**2 /2 * (slope_rem - slope)/buffer + slope*(x - x1) + min(second_lin(x1), xM)
    def third_lin(x):
        return xM - slope_rem*(xM - x)
    u = np.where(x < x0 - buffer, first_lin(x),
                 np.where(x < x0, left_para(x),
                          np.where(x < x1, second_lin(x),
                                   np.where(x < x1 + buffer, right_para(x),
                                            third_lin(x)))))
    return u
# discretised x-coords of o_x (for y-coord: use y_p)
x_o = np.linspace(0, a, N+1) # [m]
# discretised y-coords of o_y (for x-coord: use x_p)
y_o = np.linspace(0, b, M+1) # [m]
# discretised x- and y-coords of p
x_p = (x_o[1:] + x_o[:-1])/2
y_p = (y_o[1:] + y_o[:-1])/2

# change coordinates to get more points around wedge
f, buffer = 0.5, 6
#x_o = coordTransform(x_o, x0=7, x1=11, f=f)
#x_p = coordTransform(x_p, x0=7, x1=11, f=f)
# x_o = parabolicCoordTransform(x_o, x0=7, x1=11, f=f, buffer=buffer)
# x_p = parabolicCoordTransform(x_p, x0=7, x1=11, f=f, buffer=buffer)
#y_o = coordTransform(y_o, x0=0, x1=6, f=f)
#y_p = coordTransform(y_p, x0=0, x1=6, f=f)
# y_o = parabolicCoordTransform(y_o, x0=0, x1=6, f=f, buffer=buffer)
# y_p = parabolicCoordTransform(y_p, x0=0, x1=6, f=f, buffer=buffer)
if refinement:
    plt.plot(x_o, np.arange(N+1),    ".", label="x$_o$")
    plt.plot(y_o, np.arange(M+1),    ".", label="y$_o$")
    plt.plot(x_p, np.arange(N) + .5, ".", label="x$_p$")
    plt.plot(y_p, np.arange(M) + .5, ".", label="y$_p$")
    plt.xlabel("coordinate (m)")
    plt.ylabel("discretisation point i")
    plt.title("coordinate transformation")
    plt.legend()
    plt.show()

# spatial step between x-coord of o_x
dx_o = (x_o[1:] - x_o[:-1]).reshape((1,-1)) # [m]
# spatial step between y-coord of o_y
dy_o = (y_o[1:] - y_o[:-1]).reshape((-1,1)) # [m]
# spatials steps between coords of p
dx_p = (x_p[1:] - x_p[:-1]).reshape((1,-1)) # [m]
dy_p = (y_p[1:] - y_p[:-1]).reshape((-1,1)) # [m]
if refinement:
    plt.plot(np.arange(N) + 0.5, dx_o[0,:], ".", label="dx$_o$")
    plt.plot(np.arange(M) + 0.5, dy_o[:,0], ".", label="dy$_o$")
    plt.plot(np.arange(N-1) + 1, dx_p[0,:], ".", label="dx$_p$")
    plt.plot(np.arange(M-1) + 1, dy_p[:,0], ".", label="dy$_p$")
    plt.xlabel("n")
    plt.ylabel("(m)")
    plt.ylim(bottom=0)
    plt.title("spatial steps after transformation")
    plt.legend()
    plt.show()

#time step based on CN and stabilaty
# dt=np.sqrt(2)*0.05   #[s]
dt = 1/np.sqrt(1/np.min(dx_o)**2 + 1/np.min(dy_o)**2)/c
print(f"dt = {dt}")
T=40 #time of simulation [s]

# courant number
CN=c**2*dt**2*(1/np.min(dx_o)**2+1/np.min(dy_o)**2) # [s/m]
# since dx_p is mean of 2 neighbouring dx_o's it is always bigger than min of dx_o
print(f"CN = {CN} < 1?")




#constructing mesh
ny, nx = M,N
x = x_p  #(N,1) matrix
y = y_p  #(M,1) matrix
X, Y = np.meshgrid(x, y)   #(M,N) matrix both

#kappa for x-direction
F1=10*(((X-(a-d))/d)**5 *np.heaviside(X-(a-d),1)-((X-d)/d)**5*np.heaviside(-X+d,1)) #(M,N) matrix
if PML:
    #plotting
    plt.pcolormesh(X, Y, F1, shading='auto', cmap='viridis')
    plt.colorbar(label='f(x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('f(x, y) with pcolormesh')
    plt.show()

#kappa for y-direction without floor
F2=10*(((Y-(b-d))/d)**5 *np.heaviside(Y-(b-d),1))
if PML:
    #plotting
    plt.pcolormesh(X, Y, F2, shading='auto', cmap='viridis')
    plt.colorbar(label='f(x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('f(x, y) with pcolormesh')
    plt.show()

#kappa for y-direction with floor
F3=10*(((Y-(b-d))/d)**5 *np.heaviside(Y-(b-d),1)-((Y-d)/d)**5*np.heaviside(-Y+d,1))

if PML:

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

dx=a/N
dy=b/M

print("CN fr fr",c**2*dt**2*(1/dx**2+1/dy**2))

# Special interpolation (x[i] must lie between x0[i-1] and x0[i+1]).
# Quadratic interpolation in body,
# linear at edges
def interp(x, x0, y0):
    y = np.zeros(y0.shape)
    x1, y1 = x0[:-2],  y0[:-2]
    x2, y2 = x0[1:-1], y0[1:-1]
    x3, y3 = x0[2:],   y0[2:]
    tmp = (y2 - y1)/(x2 - x1)
    a = (y3 - y1 - (x3 - x1)*tmp)/(x3 - x1)/(x3-x2)
    b = tmp - (x2 + x1)*a
    c = y1 - x1*tmp + x1*x2*a
    y[1:-1] = (a*x[1:-1] + b)*x[1:-1] + c
    # linear interpolation for edges
    y[0] = (y0[1] - y0[0])/(x0[1] - x0[0]) * (x[0] - x0[0]) + y0[0]
    y[-1] = (y0[-1] - y0[-2])/(x0[-1] - x0[-2]) * (x[-1] - x0[-2]) + y0[-2]
    return y

def dp_d(p):
    dp_dx = (p[:,1:] - p[:,:-1])/dx_p
    dp_dy = (p[1:,:] - p[:-1,:])/dy_p

    x_dp_dx = ((x_p[1:] + x_p[:-1])/2).reshape((-1,1))
    # y_dp_dx = y_p
    # x_dp_dy = x_p
    y_dp_dy = ((y_p[1:] + y_p[:-1])/2).reshape((-1,1))

    dp_dx = interp(x_o[1:-1].reshape((-1,1)), x_dp_dx, dp_dx.T).T
    dp_dy = interp(y_o[1:-1].reshape((-1,1)), y_dp_dy, dp_dy)
    return dp_dx, dp_dy

def do_d(ox, oy):
    dox_dx = (ox[:,1:] - ox[:,:-1])/dx_o
    doy_dy = (oy[1:,:] - oy[:-1,:])/dy_o

    x_dox_dx = ((x_o[1:] + x_o[:-1])/2).reshape((-1,1))
    y_doy_dy = ((y_o[1:] + y_o[:-1])/2).reshape((-1,1))

    dox_dx = interp(x_p.reshape((-1,1)), x_dox_dx, dox_dx.T).T
    doy_dy = interp(y_p.reshape((-1,1)), y_doy_dy, doy_dy)
    return dox_dx, doy_dy

#simulation in empty universe
def empty(px,py,ox,oy,F1,F3,K,xs,ys,co):


    #fix dimensions/construct the final kappa's (sligth alteration to dimension so they would fit in the equations)
    k1_o=F1[:,:-1]
    k1_p=F1
    k2_o=F3[1:,:]
    k2_p=F3

    #source index
    i_s = np.argmin(np.abs(y_p - ys))
    j_s = np.argmin(np.abs(x_p - xs))

    #observation
    observations=[]
    for i in range(len(co)):
        observations.append([])


    #solving scheme
    for i in range(K-1):

        #construct total p
        p=px[i]+py[i]

        dp_dx, dp_dy = dp_d(p)

        #two equations for o
        ox[i+1,:,1:-1]=((1-k1_o*dt/2)/(1+k1_o*dt/2))*ox[i,:,1:-1]-dt/(1+k1_o*dt/2)*dp_dx
        oy[i+1,1:-1,:]=((1-k2_o*dt/2)/(1+k2_o*dt/2))*oy[i,1:-1,:]-dt/(1+k2_o*dt/2)*dp_dy

        dox_dx, doy_dy = do_d(ox[i+1], oy[i+1])

        #two equations for p
        px[i+1]=((1-k1_p*dt/2)/(1+k1_p*dt/2))*px[i]-(c**2/(1+k1_p*dt/2))*dt*dox_dx
        py[i+1]=((1-k2_p*dt/2)/(1+k2_p*dt/2))*py[i]-(c**2/(1+k2_p*dt/2))*dt*doy_dy

        

        # adding source
        px[i+1,i_s,j_s]+=Ps(i*dt)/2
        py[i+1,i_s,j_s]+=Ps(i*dt)/2
        #observing:
        for ind,r in enumerate(co):
            i_obs = np.argmin(np.abs(x_p - r[1]))
            j_obs = np.argmin(np.abs(y_p - r[0]))
            observations[ind].append(px[i+1,i_obs,j_obs]+py[i+1,i_obs,j_obs])
    return px+py,ox,oy,observations

#simulation with floor and wall
def wall(px,py,ox,oy,F1,F2,K,xs,ys,co):


    #fix dimensions/construct the final kappa's (sligth alteration to dimension so they would fit in the equations)
    k1_o=F1[:,:-1] 
    k1_p=F1
    k2_o=F2[1:,:]
    k2_p=F2
    
    #source index
    i_s = np.argmin(np.abs(y_p - ys))
    j_s = np.argmin(np.abs(x_p - xs))
  
    #observation
    observations=[]
    for i in range(len(co)):
        observations.append([])



    #solving scheme
    for i in range(K-1):

        wall_left=int(wl/a*(N+1))
        ceiling=int(wh/b*(M+1))


        #construct total p
        p=px[i]+py[i]
        dp_dx, dp_dy = dp_d(p)
        #two equations for o
        ox[i+1,:,1:-1]=((1-k1_o*dt/2)/(1+k1_o*dt/2))*ox[i,:,1:-1]-dt/(1+k1_o*dt/2)*dp_dx
        oy[i+1,1:-1,:]=((1-k2_o*dt/2)/(1+k2_o*dt/2))*oy[i,1:-1,:]-dt/(1+k2_o*dt/2)*dp_dy

        #aplying boundry conditions
        ox[i+1,:ceiling,wall_left]=0

        dox_dx, doy_dy = do_d(ox[i+1], oy[i+1])

        #two equations for p
        px[i+1]=((1-k1_p*dt/2)/(1+k1_p*dt/2))*px[i]-(c**2/(1+k1_p*dt/2))*dt*dox_dx
        py[i+1]=((1-k2_p*dt/2)/(1+k2_p*dt/2))*py[i]-(c**2/(1+k2_p*dt/2))*dt*doy_dy


        # adding source
        px[i+1,i_s,j_s]+=Ps(i*dt)/2
        py[i+1,i_s,j_s]+=Ps(i*dt)/2

        #observing:
        for ind,r in enumerate(co):
            i_obs = np.argmin(np.abs(x_p - r[1]))
            j_obs = np.argmin(np.abs(y_p - r[0]))
            observations[ind].append(px[i+1,i_obs,j_obs]+py[i+1,i_obs,j_obs])
    return px+py,ox,oy,observations

#simulation with floor and rectangle
def rectangle(px,py,ox,oy,F1,F2,K,Z,xs,ys,co):
    #fix dimensions/construct the final kappa's (sligth alteration to dimension so they would fit in the equations)
    k1_o=F1[:,:-1]
    k1_p=F1
    k2_o=F2[1:,:]
    k2_p=F2

    #source index
    i_s = np.argmin(np.abs(y_p - ys))
    j_s = np.argmin(np.abs(x_p - xs))


    #observation
    observations=[]
    for i in range(len(co)):
        observations.append([])

    
    #determie obstacle indexes
    wall_left=int(wl/a*(N+1))
    wall_right=int(wr/a*(N+1))
    ceiling=int(wh/b*(M+1))

    #solving scheme
    for i in range(K-1):
        #construct total p
        p=px[i]+py[i]
        dp_dx, dp_dy = dp_d(p)
        #two equations for o
        ox[i+1,:,1:-1]=((1-k1_o*dt/2)/(1+k1_o*dt/2))*ox[i,:,1:-1]-dt/(1+k1_o*dt/2)*dp_dx
        oy[i+1,1:-1,:]=((1-k2_o*dt/2)/(1+k2_o*dt/2))*oy[i,1:-1,:]-dt/(1+k2_o*dt/2)*dp_dy

        #aplying boundry conditions
        ox[i+1,:ceiling,wall_left]=((1-dt*Z/dx)*ox[i,:ceiling,wall_left]+2*dt/dx*p[:ceiling,wall_left-1])/(1+Z*dt/dx)
        ox[i+1,:ceiling,wall_right]=((1-dt*Z/dx)*ox[i,:ceiling,wall_right]-2*dt/dx*p[:ceiling,wall_right])/(1+Z*dt/dx)
        oy[i+1,ceiling,wall_left:wall_right]=((1-dt*Z/dy)*oy[i,ceiling,wall_left:wall_right]-2*dt/dy*p[ceiling,wall_left:wall_right])/(1+Z*dt/dy)

        dox_dx, doy_dy = do_d(ox[i+1], oy[i+1])

        #two equations for p
        px[i+1]=((1-k1_p*dt/2)/(1+k1_p*dt/2))*px[i]-(c**2/(1+k1_p*dt/2))*dt*dox_dx
        py[i+1]=((1-k2_p*dt/2)/(1+k2_p*dt/2))*py[i]-(c**2/(1+k2_p*dt/2))*dt*doy_dy

        #removing inside obstacle ->not sure if neccecary
        #px[i+1,:ceiling,wall_left:wall_right]=0
        #py[i+1,:ceiling,wall_left:wall_right]=0

        # adding source
        px[i+1,i_s,j_s]+=Ps(i*dt)/2
        py[i+1,i_s,j_s]+=Ps(i*dt)/2

        #observing:
        for ind,r in enumerate(co):
            i_obs = np.argmin(np.abs(x_p - r[1]))
            j_obs = np.argmin(np.abs(y_p - r[0]))
            observations[ind].append(px[i+1,i_obs,j_obs]+py[i+1,i_obs,j_obs])
    return px+py,ox,oy,observations

# simulation with triangle
def triangle(px,py,ox,oy,F1,F2,K,xs,ys,co):
    #fix dimensions/construct the final kappa's (sligth alteration to dimension so they would fit in the equations)
    k1_o=F1[:,:-1]
    k1_p=F1
    k2_o=F2[1:,:]
    k2_p=F2

    #source index
    i_s = np.argmin(np.abs(y_p - ys))
    j_s = np.argmin(np.abs(x_p - xs))

    #observation
    observations=[]
    for i in range(len(co)):
        observations.append([])

    
    
    #determie obstacle indexes
    wall_left=int(wl/a*(N+1))
    wall_right=int(wr/a*(N+1))
    ceiling=int(wh/b*(M+1))
    print("-"*20)
    print(ceiling)
    print((wall_right-wall_left)*2)

    #solving scheme
    for i in range(K-1):
        #construct total p
        p=px[i]+py[i]
        dp_dx, dp_dy = dp_d(p)
        #two equations for o
        ox[i+1,:,1:-1]=((1-k1_o*dt/2)/(1+k1_o*dt/2))*ox[i,:,1:-1]-dt/(1+k1_o*dt/2)*dp_dx
        oy[i+1,1:-1,:]=((1-k2_o*dt/2)/(1+k2_o*dt/2))*oy[i,1:-1,:]-dt/(1+k2_o*dt/2)*dp_dy

        #aplying boundry conditions
        midle=int((wall_right-wall_left)/2)
        
        for index in range(midle+1):
            oy[i+1,4*index,index+wall_left]=((1-Z*(dt/dy_o[0,0]))*ox[i,4*index,index+wall_left]-2*(dt/dx_o[0,0])*p[4*index,index+wall_left])/(1+Z*(dt/dy_o[0,0]))
            ox[i+1,4*index,index+wall_left+1]=((1-Z*(dt/dx_o[0,0]))*ox[i,4*index,index+wall_left+1]+2*(dt/dx_o[0,0])*p[4*index,index+wall_left+1-1])/(1+Z*(dt/dx_o[0,0]))
            ox[i+1,4*index+1,index+wall_left+1]=((1-Z*(dt/dx_o[0,0]))*ox[i,4*index+1,index+wall_left+1]+2*(dt/dx_o[0,0])*p[4*index+1,index+wall_left+1-1])/(1+Z*(dt/dx_o[0,0]))
            ox[i+1,4*index+2,index+wall_left+1]=((1-Z*(dt/dx_o[0,0]))*ox[i,4*index+2,index+wall_left+1]+2*(dt/dx_o[0,0])*p[4*index+2,index+wall_left+1-1])/(1+Z*(dt/dx_o[0,0]))
            ox[i+1,4*index+3,index+wall_left+1]=((1-Z*(dt/dx_o[0,0]))*ox[i,4*index+3,index+wall_left+1]+2*(dt/dx_o[0,0])*p[4*index+3,index+wall_left+1-1])/(1+Z*(dt/dx_o[0,0]))
            up_midle=4*midle+3
            start_midle=midle+wall_left+1
            oy[i+1,up_midle-4*index,index+start_midle]=((1-Z*(dt/dy_o[0,0]))*oy[i,up_midle-4*index,index+start_midle]-2*(dt/dx_o[0,0])*p[up_midle-4*index,index+start_midle])/(1+Z*(dt/dy_o[0,0]))
            ox[i+1,up_midle-4*index-1,index+start_midle+1]=((1-Z*(dt/dx_o[0,0]))*ox[i,up_midle-4*index-1,index+start_midle+1]-2*(dt/dx_o[0,0])*p[up_midle-4*index-1,index+start_midle+1])/(1+Z*(dt/dx_o[0,0]))
            ox[i+1,up_midle-4*index-1-1,index+start_midle+1]=((1-Z*(dt/dx_o[0,0]))*ox[i,up_midle-4*index-1-1,index+start_midle+1]-2*(dt/dx_o[0,0])*p[up_midle-4*index-1-1,index+start_midle+1])/(1+Z*(dt/dx_o[0,0]))
            ox[i+1,up_midle-4*index-2-1,index+start_midle+1]=((1-Z*(dt/dx_o[0,0]))*ox[i,up_midle-4*index-2-1,index+start_midle+1]-2*(dt/dx_o[0,0])*p[up_midle-4*index-2-1,index+start_midle+1])/(1+Z*(dt/dx_o[0,0]))
            ox[i+1,up_midle-4*index-3-1,index+start_midle+1]=((1-Z*(dt/dx_o[0,0]))*ox[i,up_midle-4*index-3-1,index+start_midle+1]-2*(dt/dx_o[0,0])*p[up_midle-4*index-3-1,index+start_midle+1])/(1+Z*(dt/dx_o[0,0]))
        

        dox_dx, doy_dy = do_d(ox[i+1], oy[i+1])

        #two equations for p
        px[i+1]=((1-k1_p*dt/2)/(1+k1_p*dt/2))*px[i]-(c**2/(1+k1_p*dt/2))*dt*dox_dx
        py[i+1]=((1-k2_p*dt/2)/(1+k2_p*dt/2))*py[i]-(c**2/(1+k2_p*dt/2))*dt*doy_dy

        # adding source
        px[i+1,i_s,j_s]+=Ps(i*dt)/2
        py[i+1,i_s,j_s]+=Ps(i*dt)/2

        #observing:
        for ind,r in enumerate(co):
            i_obs = np.argmin(np.abs(x_p - r[1]))
            j_obs = np.argmin(np.abs(y_p - r[0]))
            observations[ind].append(px[i+1,i_obs,j_obs]+py[i+1,i_obs,j_obs])
    return px+py,ox,oy,observations


# p,ox,oy,obs=empty(px,py,ox,oy,F1,F3,K,xs,ys,coordinates1)
# p,ox,oy,obs=empty(px,py,ox,oy,F1,F3,K,xs,ys,coordinates2)
#p,ox,oy,obs=wall(px,py,ox,oy,F1,F2,K,xs,ys,coordinates1)
#p,ox,oy,obs=rectangle(px,py,ox,oy,F1,F2,K,Z,xs,ys,coordinates2)
p,ox,oy,obs=triangle(px,py,ox,oy,F1,F2,K,xs,ys,coordinates2)

print(f"max |p| = {np.max(np.abs(p))}")

#animate function


def animate_heatmap(frames,xs,ys, coords,
                    interval=100, cmap='viridis',
                    save_path=None, fps=10):
    """
    Animate a sequence of 2D numpy arrays (frames) as a heatmap,
    with fixed points overlaid.

    Parameters
    ----------
    frames : np.ndarray
        3D array of shape (num_frames, height, width)
    coords : list of [x, y]
        Points to overlay on the heatmap
    x_p, y_p : 2D arrays
        Grid coordinates for pcolormesh
    """

    if frames.ndim != 3:
        raise ValueError("Input 'frames' must be a 3D NumPy array.")

    num_frames, height, width = frames.shape

    fig, ax = plt.subplots()

    # Initial heatmap
    cm = ax.pcolormesh(
        x_p, y_p, frames[0],
        shading='nearest',
        cmap=cmap
    )
    plt.colorbar(cm, ax=ax)
    ax.set_aspect('equal', 'box')

    # Convert coords to arrays
    
    coords = np.asarray(coords)
    X, Y = coords[:, 0], coords[:, 1]

    # Overlay points
    scatterO = ax.scatter(
        X, Y,
        c='red',
        s=40,
        marker='o',
        edgecolors='black',
        zorder=3,
        label='Points'
    )
    scatterS = ax.scatter(
        xs, ys,
        c='green',
        s=40,
        marker='o',
        edgecolors='black',
        zorder=3,
        label='Points'
    )
    ax.legend()

    def update(frame_idx):
        cm.set_array(frames[frame_idx].ravel())
        ax.set_title(f"Frame {frame_idx + 1}/{num_frames}")
        return cm, scatterO,scatterS

    ani = animation.FuncAnimation(
        fig, update,
        frames=num_frames,
        interval=interval,
        blit=True
    )

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


animate_heatmap(p,xs,ys,coordinates2)

t = dt*np.arange(0, K-1)

plt.plot(t,obs[1])
plt.show()

