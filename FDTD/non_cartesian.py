from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt

# TODO:
# - PML (left, up and right)
# - finite Z
# - free field
# - 

# Where normally one uses x and y direction orthogonal to eachother
# Use here u and v which make an angle theta
# e_u = e_x                                  e_x = e_u
# e_v = e_x * cos theta + e_y * sin theta    e_y = (e_v - e_u * cos theta) / sin theta
# introduce vector e_w which is orthogonal to e_v
# e_w = e_x * sin theta - e_y * cos theta  
theta = np.arctan(2 / (1/2)) # [rad]
def to_xy(u, v):
    return u + v*np.cos(theta), v*np.sin(theta)

c = 1 # wavespeed [m/s]

d = 6 # characteristic distance [m]

us = d                               # u-coordinate of source [m]
vs = d/10                            # v-coordinate of source [m]
Ps = lambda t: 4*np.exp(-(t - 3)**2) # source

# returns True if x and y lie in the triangle
def triangle(x, y):
    xs, _ = to_xy(us, vs) # x-coord of source [m]
    # to be in triangle is same as being beneath two lines
    return (y < 4*(x - xs - d)) & (y < -4 * (x - xs - 2*d))

# Currently using a padding of d around important stuff (temporary)
a = 5*d + 2*d                 # width in u-direction  [m]
b = (2*d + 2*d)/np.sin(theta) # height in v-direction [m]
T = 2*b/c                     # timespan              [s]

# Courant Number
CN  = 0.9

du = 0.2                                             # step in u-direction [m]
dv = du/np.sin(theta)                                # step in v-direction [m]
dt = CN/np.sqrt(1/du**2 + 1/(dv*np.sin(theta))**2)/c # timestep            [s]
print(f"du x dv x dt = {du} x {dv} x {dt}")

nu = int(np.ceil(a/du)) - 1 # size in u-direction [1]
nv = int(np.ceil(b/dv)) - 1 # size in v-direction [1]
nt = int(np.ceil(T/dt))     # size in time        [1]
print(f"nx x nv x nt = {nu} x {nv} x {nt}")

pw = np.zeros((nu,     nv,     nt)) # p at discrete u, v and t points
pv = np.zeros((nu,     nv,     nt)) # p at discrete u, v and t points
ow = np.zeros((nu + 1,     nv, nt)) # w component of o
ov = np.zeros((nu, nv + 1,     nt)) # v component of o

u_p = np.arange(du/2, a - du/2, du).reshape((nu, 1,  1))  # u-coordinates of discretized p field [m]
v_p = np.arange(dv/2, b - dv/2, dv).reshape((1,  nv, 1))  # v-coordinates                        [m]
t_p = np.arange(0, T, dt).reshape((1,  1,  nt))           # t-coordinates                        [s]

u_ow = np.arange(0, a, du).reshape((nu + 1, 1, 1))
v_ow = v_p
t_ow = np.arange(dt/2, T + dt/2, dt).reshape((1, 1, nt))

u_ov = u_p
v_ov = np.arange(0, b, dv).reshape((1, nv + 1, 1))
t_ov = t_ow

plt.plot(t_p.reshape(-1), Ps(t_p.reshape(-1)))
plt.title('source')
plt.show()

# damping coefficients PML
m = 5       # damping exp         [1]
kappaM = 10 # maximum kappa value [1/s]
d_PML = d   # thickness of PML    [m]
def kappa(x):
    return kappaM * ((d_PML - x)/d_PML)**m * np.heaviside(d_PML - x, 1)
kappa_horizontal_p  = kappa(u_p .reshape((-1,1))) + kappa(a - u_p .reshape((-1,1)))
kappa_horizontal_ow = kappa(u_ow.reshape((-1,1))) + kappa(a - u_ow.reshape((-1,1)))
kappa_up_p  = kappa(b - v_p .reshape((1,-1)))
kappa_up_ov = kappa(b - v_ov.reshape((1,-1)))
kappa_pw = kappa_horizontal_p
kappa_pv = kappa_up_p
kappa_ow = kappa_horizontal_ow[1:-1,:]
kappa_ov = kappa_up_ov[:,1:-1]

plt.plot(u_p.reshape(-1), kappa_horizontal_p.reshape(-1))
plt.xlabel("u (m)")
plt.ylabel("\\kappa")
plt.title("horizontale dempingscoefficient")
plt.show()

TRIANGLE_ow = triangle(*to_xy(u_ow, v_ow)).reshape((nu + 1, nv))
TRIANGLE_ov = triangle(*to_xy(u_ov, v_ov)).reshape((nu,     nv + 1))

# scheme
for l in range(1, nt):
    # total p (e_v and e_w are orthogonal) at preceding timestep
    p = pv[:,:,l-1] + pw[:,:,l-1]

    dp_du = (p[1:,: ] - p[:-1,:  ])/du
    # e_v . nabla p = (e_x * cos theta + e_y * sin theta) . nabla p = dp/dx cos theta + dp/dy sin theta
    dp_dv = (p[:, 1:] - p[:,  :-1])/dv

    dp_dx = dp_du

    tmp = np.gradient(p[:,:], axis=1) # dp/dv but at original points
    # e_w . nabla p = (e_x * sin theta - e_y * cos theta) . nabla p = dp/dx sin theta - dp/dy cos theta
    #               = dp/dx sin theta - (dp/dv - dp/dx cos theta) cos theta / sin theta
    #               = (dp/dx - dp/dv cos theta) / sin theta
    dp_dw = (dp_dx - (tmp[1:,:] + tmp[:-1,:])/2*np.cos(theta)) / np.sin(theta)

    # update o
    ow[1:-1,:,l] = ((1 - dt*kappa_ow/2)*ow[1:-1,:,l-1] - dt*dp_dw)/(1 + dt*kappa_ow/2)
    ov[:,1:-1,l] = ((1 - dt*kappa_ov/2)*ov[:,1:-1,l-1] - dt*dp_dv)/(1 + dt*kappa_ov/2)

    # BC
    ow[TRIANGLE_ow,l] = 0

    dow_du = (ow[1:,:,l] - ow[:-1,:,l])/du
    dow_dv = (ow[:,1:,l] - ow[:,:-1,l])/dv
    dov_dv = (ov[:,1:,l] - ov[:,:-1,l])/dv

    dow_dx = dow_du

    tmp = np.gradient(ow[:,:,l], axis=1) # dow/dv but at original points
    # e_w . nabla ow = (e_x * sin theta - e_y * cos theta) . nabla ow = dow/dx sin theta - dow/dy cos theta
    dow_dw = (dow_dx - (tmp[1:,:] + tmp[:-1,:])/2*np.cos(theta)) / np.sin(theta)

    div_o = dow_dw + dov_dv

    # update p
    pw[:,:,l] = ((1 - dt*kappa_pw/2)*pw[:,:,l-1] - c**2*dt*dow_dw)/(1 + dt*kappa_pw/2)
    pv[:,:,l] = ((1 - dt*kappa_pv/2)*pv[:,:,l-1] - c**2*dt*dov_dv)/(1 + dt*kappa_pv/2)

    # sources
    pw[int(us/du), int(vs/dv), l] += Ps(l*dt)/2
    pv[int(us/du), int(vs/dv), l] += Ps(l*dt)/2

U, V = np.meshgrid(u_p, v_p, indexing='ij')
X, Y = to_xy(U, V)

cm = plt.pcolormesh(X, Y, triangle(X, Y))
plt.colorbar(cm)
plt.show()

p = pv + pw

fig, ax = plt.subplots()
cm = ax.pcolormesh(X, Y, p[:,:,0])

def update(frame):
    cm.set_array(p[:,:,frame])
    return (cm,)
ani = animation.FuncAnimation(fig=fig, func=update, frames=nt, repeat=True)
ax.set_aspect('equal', 'box')
# ani.save('tmp.gif', writer='pillow')
fig.colorbar(cm)
plt.show()