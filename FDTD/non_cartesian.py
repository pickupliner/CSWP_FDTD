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
# e_u = e_x
# e_v = e_x * cos theta + e_y * sin theta
theta = np.arctan(2 / (1/2)) # [rad]
def to_xy(u, v):
    return u + v*np.cos(theta), v*np.sin(theta)

c = 1 # wavespeed [m/s]

d = 6 # characteristic distance [m]

us = d                                   # u-coordinate of source [m]
vs = d/10                                # v-coordinate of source [m]
Ps = lambda t: 4*np.exp(-(t/2 - 2.5)**2) # source

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
CN  = 1.0

du = 0.2                                             # step in u-direction [m]
dv = du/np.sin(theta)                                # step in v-direction [m]
dt = CN/np.sqrt(1/du**2 + 1/(dv*np.sin(theta))**2)/c # timestep            [s]
print(f"du x dv x dt = {du} x {dv} x {dt}")

nu = int(np.ceil(a/du)) - 1 # size in u-direction [1]
nv = int(np.ceil(b/dv)) - 1 # size in v-direction [1]
nt = int(np.ceil(T/dt))     # size in time        [1]
print(f"nx x nv x nt = {nu} x {nv} x {nt}")

p  = np.zeros((nu,     nv,     nt)) # p at discrete u, v and t points
ox = np.zeros((nu + 1,     nv, nt)) # x component of o
oy = np.zeros((nu, nv + 1,     nt)) # y component of o

u_p = np.arange(du/2, a - du/2, du).reshape((nu, 1,  1)) # u-coordinates of discretized p field [m]
v_p = np.arange(dv/2, b - dv/2, dv).reshape((1,  nv, 1)) # v-coordinates                        [m]
t_p = np.arange(0, T, dt).reshape((1,  1,  nt))          # t-coordinates                        [s]

u_ox = np.arange(0, a, du).reshape((nu + 1, 1, 1))
v_ox = v_p
t_ox = np.arange(dt/2, T + dt/2, dt).reshape((1, 1, nt))

u_oy = u_p
v_oy = np.arange(0, b, dv).reshape((1, nv + 1, 1))
t_oy = t_ox

plt.plot(t_p.reshape(-1), Ps(t_p.reshape(-1)))
plt.title('source')
plt.show()

TRIANGLE_ox = triangle(*to_xy(u_ox, v_ox)).reshape((nu + 1, nv))
TRIANGLE_oy = triangle(*to_xy(u_oy, v_oy)).reshape((nu,     nv + 1))

# scheme
for l in range(1, nt):
    dp_du = (p[1:,:,l-1] - p[:-1,:,l-1])/du
    # e_v . nabla p = (e_x * cos theta + e_y * sin theta) . nabla = dp/dx cos theta + dp/dy sin theta
    dp_dv = (p[:, 1:,l-1] - p[:,:-1,l-1])/dv

    dp_dx = dp_du
    tmp = np.gradient(p[:,:,l-1], axis=1) # dp/dv but at original points
    dp_dy = (dp_dv - (tmp[:,1:] + tmp[:,:-1])/2*np.cos(theta))/np.sin(theta)

    # update o
    ox[1:-1,:,l] = ox[1:-1,:,l-1] - dt*dp_dx
    oy[:,1:-1,l] = oy[:,1:-1,l-1] - dt*dp_dy

    # BC
    ox[TRIANGLE_ox,l] = 0
    oy[TRIANGLE_oy,l] = 0

    dox_du = (ox[1:,:,l] - ox[:-1,:,l])/du
    doy_du = (oy[1:,:,l] - oy[:-1,:,l])/du
    doy_dv = (oy[:,1:,l] - oy[:,:-1,l])/dv

    dox_dx = dox_du
    doy_dx = doy_du
    tmp = np.gradient(oy[:,:,l], axis=0) # doy/du but at original points
    doy_dy = (doy_dv - (tmp[:,1:] + tmp[:,:-1])/2*np.cos(theta))/np.sin(theta)

    div_o = dox_dx + doy_dy

    # update p
    p[:,:,l] = p[:,:,l-1] - c**2 * dt * div_o

    # sources
    p[int(us/du), int(vs/dv), l] += Ps(l*dt)

U, V = np.meshgrid(u_p, v_p, indexing='ij')
X, Y = to_xy(U, V)

cm = plt.pcolormesh(X, Y, triangle(X, Y))
plt.colorbar(cm)
plt.show()

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