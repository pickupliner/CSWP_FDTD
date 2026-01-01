from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt

# Where normally one uses x and y direction orthogonal to eachother
# Use here u and v which make an angle theta
# e_u = e_x                                  e_x = e_u
# e_v = e_x * cos theta + e_y * sin theta    e_y = (e_v - e_u * cos theta) / sin theta
# introduce vector e_w which is orthogonal to e_v
# e_w = e_x * sin theta - e_y * cos theta  
theta = np.arctan(2 / (1/2)) # [rad]
def to_xy(u, v):
    return u + v*np.cos(theta), v*np.sin(theta)
def to_uv(x, y):
    return x - y/np.tan(theta), y/np.sin(theta)

c = 1 # wavespeed [m/s]

d = 5.6 # characteristic distance [m]

# use a padding of d around important stuff
a = 5*d + 2*d                 # width in u-direction  [m]
b = (2*d + 2*d)/np.sin(theta) # height in v-direction [m]
T = 6*b/c                     # timespan              [s]

# Courant Number
CN  = 0.8

du = 0.2                                             # step in u-direction [m]
dv = du/np.sin(theta)                                # step in v-direction [m]
dt = np.sqrt(CN)/np.sqrt(1/(du + dv*np.cos(theta))**2 + 1/(dv*np.sin(theta))**2)/c # timestep            [s]
print(f"du x dv x dt = {du} x {dv} x {dt}")

# PML parameters
m = 5             # damping exp         [1]
kappaM = 10       # maximum kappa value [1/s]
d_PML = 4 # 20*dv # thickness of PML    [m]
def kappa(x):
    return kappaM * ((d_PML - x)/d_PML)**m * np.heaviside(d_PML - x, 1)

# returns True if x and y lie in the triangle
# to be in triangle is same as being beneath two lines
# left triangle has accurate BC of o_n = 0, right just all o zero
def left_triangle(x, y):
    return (y < 4*(x - xs - d)) & (x <= xs + 3*d/2)
def right_triangle(x, y):
    return (y < -4 * (x - xs - 2*d)) & (x > xs + 3*d/2)
# returns True if x and y are in the ground
def ground(x, y):
    return (y <= 0) & (np.ones_like(x) != 0)

nu = int(np.ceil(a/du)) - 1         # size in u-direction [1]
nv = int(np.ceil((b+d_PML)/dv)) - 1 # size in v-direction [1]
nt = int(np.ceil(T/dt))             # size in time        [1]
print(f"nx x nv x nt = {nu} x {nv} x {nt}")

u_p = np.arange(du/2,         a - du/2, du).reshape((nu, 1,  1))  # u-coordinates of discretized p field [m]
v_p = np.arange(dv/2 - d_PML, b - dv/2, dv).reshape((1,  nv, 1))  # v-coordinates                        [m]
t_p = np.arange(0,            T,        dt).reshape((1,  1,  nt)) # t-coordinates                        [s]

u_ow = np.arange(0,    a,        du).reshape((nu + 1, 1, 1))
v_ow = v_p
t_ow = np.arange(dt/2, T + dt/2, dt).reshape((1, 1, nt))

u_ov = u_p
v_ov = np.arange(-d_PML, b, dv).reshape((1, nv + 1, 1))
t_ov = t_ow

# plt.plot(*to_xy(u_p[:2,:,0],  v_p[:,:2,0]),  'o', color='black')
# x, y = to_xy(u_ow[:3,:,0], v_ow[:,:2,0])
# dx, dy = du/8*np.sin(theta), -du/8*np.cos(theta)
# for (x, y) in zip(x.reshape(-1), y.reshape(-1)):
#     plt.arrow(x - dx, y - dy, 2*dx, 2*dy, width=0.004, fc='black')
# x, y = to_xy(u_ov[:2,:,0], v_ov[:,:3,0])
# dx, dy = du/8*np.cos(theta), du/8*np.sin(theta)
# for (x, y) in zip(x.reshape(-1), y.reshape(-1)):
#     plt.arrow(x - dx, y - dy, 2*dx, 2*dy, width=0.004, fc='black')
# plt.gca().set_aspect('equal', 'box')
# plt.xlabel("x (m)")
# plt.ylabel("y (m)")
# plt.show()

# source
us = d                               # u-coordinate of source [m]
vs = d/10                            # v-coordinate of source [m]
xs, ys = to_xy(us, vs)               # x- and y-coordinate    [m]
f = 4.95*c/(d*2*np.pi)               # central frequency      [Hz]
sigma = f
t0 = 18                              # center of pulse        [s]
Ps = lambda t: 10 * np.sin(2*np.pi*f*(t - t0)) * np.exp(-((t - t0)**2)*(sigma**2)) # short band around f so that kd £[0.1,10]
# Ps = lambda t: 4*np.exp(-(t - 3)**2) # source
# source indices
i_s, j_s = np.argmin(np.abs(u_p - us)), np.argmin(np.abs(v_p - vs))
plt.plot(t_p.reshape(-1), Ps(t_p.reshape(-1)))
plt.title('source')
plt.show()

# Observers
xo = [4*d, 5*d, 6*d]   # x coords of observers [m]
yo = [d/2, d/2, d/2]   # y coords of observers [m]
uo, vo = to_uv(xo, yo) # u- and v-coords       [m]
# observer indices
i_o = [np.argmin(np.abs(u_p - u)) for u in uo]
j_o = [np.argmin(np.abs(v_p - v)) for v in vo]

# damping coefficients PML
kappa_horizontal_p  = kappa(u_p .reshape((-1,1))) + kappa(a - u_p .reshape((-1,1)))
kappa_horizontal_ow = kappa(u_ow.reshape((-1,1))) + kappa(a - u_ow.reshape((-1,1)))
kappa_vertical_p    = kappa(v_p .reshape((1,-1)) + d_PML) + kappa(b - v_p .reshape((1,-1)))
kappa_vertical_ov   = kappa(v_ov.reshape((1,-1)) + d_PML) + kappa(b - v_ov.reshape((1,-1)))
kappa_pw = kappa_horizontal_p + kappa_vertical_p * np.cos(theta)
kappa_pv = 0                  + kappa_vertical_p * np.sin(theta)
kappa_ow = kappa_horizontal_ow[1:-1,:] + kappa_vertical_p          * np.cos(theta)
kappa_ov = 0                           + kappa_vertical_ov[:,1:-1] * np.sin(theta)

plt.plot(v_p.reshape(-1), kappa_vertical_p.reshape(-1))
plt.xlabel("u (m)")
plt.ylabel("\\kappa")
plt.title("verticale dempingscoefficient")
plt.show()

# PECs
LEFT_TRIANGLE_ow  = left_triangle( *to_xy(u_ow, v_ow)).reshape((nu + 1, nv))
LEFT_TRIANGLE_ov  = left_triangle( *to_xy(u_ov, v_ov)).reshape((nu,     nv + 1))
RIGHT_TRIANGLE_ow = right_triangle(*to_xy(u_ow, v_ow)).reshape((nu + 1, nv))
RIGHT_TRIANGLE_ov = right_triangle(*to_xy(u_ov, v_ov)).reshape((nu,     nv + 1))
GROUND_ow         = ground(        *to_xy(u_ow, v_ow)).reshape((nu + 1, nv))
GROUND_ov         = ground(        *to_xy(u_ov, v_ov)).reshape((nu,     nv + 1))
j_ov_ground = np.argmin(np.abs(v_ov))

def solve(with_triangle, with_ground):
    pw = np.zeros((nu,     nv,     nt)) # p at discrete u, v and t points
    pv = np.zeros((nu,     nv,     nt)) # p at discrete u, v and t points
    ow = np.zeros((nu + 1,     nv, nt)) # w component of o
    ov = np.zeros((nu, nv + 1,     nt)) # v component of o

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
        if with_triangle:
            ow[LEFT_TRIANGLE_ow,l] = 0

            ow[RIGHT_TRIANGLE_ow,l] = 0
            ov[RIGHT_TRIANGLE_ov,l] = 0
        if with_ground:
            # approximate ow at the ground interface by the one just above it
            # interpolate linearly along u
            ow_ground = (ow[1:,j_ov_ground,l] + ow[:-1,j_ov_ground,l])/2
            ov_ground = ov[:,j_ov_ground,l]

            ox_ground = ow_ground*np.sin(theta) + ov_ground*np.cos(theta)
            # leave only the x component
            ov[:,j_ov_ground,l] = ox_ground*np.cos(theta)

        dow_du = (ow[1:,:,l] - ow[:-1,:,l])/du
        # dow_dv = (ow[:,1:,l] - ow[:,:-1,l])/dv
        dov_dv = (ov[:,1:,l] - ov[:,:-1,l])/dv

        dow_dx = dow_du

        dow_dv = np.gradient(ow[:,:,l], axis=1) # dow/dv but at original points
        # e_w . nabla ow = (e_x * sin theta - e_y * cos theta) . nabla ow = dow/dx sin theta - dow/dy cos theta
        dow_dw = (dow_dx - (dow_dv[1:,:] + dow_dv[:-1,:])/2*np.cos(theta)) / np.sin(theta)

        # update p
        pw[:,:,l] = ((1 - dt*kappa_pw/2)*pw[:,:,l-1] - c**2*dt*dow_dw)/(1 + dt*kappa_pw/2)
        pv[:,:,l] = ((1 - dt*kappa_pv/2)*pv[:,:,l-1] - c**2*dt*dov_dv)/(1 + dt*kappa_pv/2)

        # remove p inside PEC
        if with_ground:
            pw[:,:j_ov_ground,l] = 0
            pv[:,:j_ov_ground,l] = 0

        # sources
        pw[i_s, j_s, l] += Ps(l*dt)/2
        pv[i_s, j_s, l] += Ps(l*dt)/2
    return (pw + pv), ow, ov

p, ow, ov = solve(with_triangle=True, with_ground=True)

# free field
p_free, ow_free, ov_free = solve(with_triangle=False, with_ground=False)

# relative field
p_rel, ow_rel, ov_rel = np.ones_like(p), np.ones_like(ow), np.ones_like(ov)
p_rel, ow_rel, ov_rel = p/p_free, ow/ow_free, ov/ov_free

U, V = np.meshgrid(u_p, v_p, indexing='ij')
X, Y = to_xy(U, V)

cm = plt.pcolormesh(X, Y, left_triangle(X, Y) | right_triangle(X, Y) | ground(X, Y))
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

# frequency domain at observer points
kd = 2*np.pi * np.fft.rfftfreq(nt, dt) / c * d

plt.plot(kd, np.abs(np.fft.rfft(Ps(t_p.reshape(-1)))))
plt.show()

for (i, j) in zip(i_o, j_o):
    P      = np.fft.rfft(p     [i, j, :])
    plt.plot(kd, np.abs(P))
plt.figure()
for (i, j) in zip(i_o, j_o):
    P_free = np.fft.rfft(p_free[i, j, :])
    plt.plot(kd, np.abs(P_free))
plt.show()

for (i, j) in zip(i_o, j_o):
    P      = np.fft.rfft(p     [i, j, :])
    P_free = np.fft.rfft(p_free[i, j, :])
    plt.plot(kd, np.abs(P/P_free))
plt.show()