import numpy as np
import scipy
import scipy.special as fns
import matplotlib.pyplot as plt
import functools

from matplotlib.animation import FuncAnimation

# Not finished yet (validation)
# Constants (N_S, ..) are placeholders
# Indices are a bit of a mess, my apologies

# indices to get x- and y-coordinate
_x, _y = 0, 1
# speed of wave
c = 3e8 # m/s
# permeability
mu = np.pi*4e-7 # H/m

# radius of cilinder
R = 1 # m

# spatial points
N_S = 32
# temporal 
N_T = 512
dt = np.pi*R/c # s
# Gauss quad order
N_G = 8

# curve

def curve(s):
    # curve(0) = curve(1)
    # s in range [0, 1]
    arg = 2*np.pi * s                             # rad
    return R*np.array([np.cos(arg), np.sin(arg)]) # m  (2, s.shape)
# (the first and last point are equal for simplicity)
curve_points = curve(np.linspace(0, 1, N_S + 1))          # m  (2, N_S + 1)
plt.plot(*curve_points)
plt.title("Geometry")
plt.show()
# tangential vectors to each segment
tangents = curve_points[:,1:] - curve_points[:,:-1]       # m  (2, N_S)
# length of each vector
l = np.linalg.norm(tangents, axis=0)                      # m  (N_S,)
# minimum of length and 2 c dt of segments
L = np.min([l, np.broadcast_to(2*c*dt, l.shape)], axis=0) # m  (N_S,)
# every segment is linearly interpolated with parameter s (0 -> 1)
# linear approximation of the curve
def rho_n(s):
    new_shape = (2, N_S,) + len(s.shape)*(1,)
    return curve_points[:,:-1].reshape(new_shape) + s*tangents.reshape(new_shape) # m  (2, N_S, s.shape)

# incident wave

t_0 = N_T/np.log2(N_T/8) * dt # s
T = c*t_0 / np.sqrt(2*np.pi) /2 # m

T1 = 20*dt

t_01 = 10*T1

def E_i(rho, t):
    gamma = 4/T * (c*(t - t_0) - rho[_x])         # 1
    return 4/T/np.sqrt(np.pi) * np.exp(-gamma**2) # 1/m
x = np.linspace(-T, T, 128).reshape((-1,1))
t = np.linspace(0, N_T*dt, N_T).reshape((1,-1))
cm = plt.pcolormesh(*np.meshgrid(x, t), E_i([x, 0], t).T)
plt.title("incoming field E$^i$")
plt.xlabel("x (m)")
plt.ylabel("t (s)")
plt.colorbar(cm)
plt.show()

# # basis functions

# # T(i)(t) = 1 if (i-1) dt < t < i dt
# def T_basis(i):
#     def T(t):
#         return np.where((-dt < t) & (t < 0), 1, 0)
#     return lambda t: T(t - i*dt)
# t = np.linspace(dt, 4*dt, 100)
# plt.plot(t, T_basis(3)(t))
# plt.title("basis function T$_3$")
# plt.xlabel("t (s)")
# plt.show()

# # f(n)(rho) = 1 if rho is on segment n
# def f(n):
#     def f(rho):
#         # if d_1 + d_2 = d_0 then rho is on the line
#         d_0 = l[n]
#         d_1 = np.linalg.norm(rho - curve_points[:,n], axis=-1)
#         d_2 = np.linalg.norm(curve_points[:,n+1] - rho, axis=-1)
#         return np.isclose(d_1 + d_2, d_0)
#     return f
# x = np.linspace(0, R, 1_000)
# y = np.linspace(0, R, 1_000)
# mesh = np.array(np.meshgrid(x, y))
# plt.pcolormesh(*mesh, f(2)(mesh.T).T)
# plt.title("basis function f$_2$")
# plt.xlabel("x (m)")
# plt.ylabel("y (m)")
# plt.show()

def F(k, rho_m, rho_prime):
    tmp = np.linalg.norm(rho_m.reshape(rho_m.shape + (1,1,1,)) - rho_prime, axis=0)/c
    a = np.max([np.broadcast_to([k*dt], tmp.shape), tmp], axis=0)
    b = np.max([np.broadcast_to([(k+1)*dt], tmp.shape), tmp], axis=0)
    return np.log((b + np.sqrt(b**2 - tmp**2))/(a + np.sqrt(a**2 - tmp**2)))

# s: -1 to 1
s, w = np.polynomial.legendre.leggauss(N_G)
# s: 0 to 1
s, w = (s.reshape((1, 1, N_G)) + 1)/2, w.reshape((1, 1, N_G))/2

# Z is called very often with the same arguments in the main for loop
# cache saves Z when called
@functools.cache
def Z(k):
    tmp = np.sqrt((2*c*dt)**2 - L**2)
    Z_0 = np.diag(-L/2/np.pi * np.log((2*c*dt + tmp)/L) - c*dt/np.pi * np.arctan(L/tmp))

    m = np.arange(N_S).reshape((N_S, 1))
    n = np.arange(N_S).reshape((1, N_S))
    tmp = np.sum(w * F(k, curve_points[:,m], rho_n(s)[:,n]), axis=-1).reshape(N_S, N_S)
    Z = -l[n]/2/np.pi * tmp
    return np.where((k==0) & (m==n), Z_0, Z)

def V(j):
    return E_i(curve_points[:,:-1], j*dt)

# I'm using GMRES cause I didn't want to think about when I'm allowed to use stuff

U = np.zeros((N_S, N_T))
# unlike in the assignment j goes from 0 to N_T-1 instead of from 1 to N_T
# so that if we call the assignment's j' then j' = j + 1
# j = 0
A = Z(0)
b = -V(0)
U[:,0], exitCode = scipy.sparse.linalg.gmres(A, b)
assert exitCode == 0 # error if equation could not be solved

for j in range(1, N_T):
    # k = np.arange(0, j-1 + 1) # k = 0, .., j-1
    def term(k):
        return np.tensordot(Z(k), U[:,j-k-1], axes=(-1,0))
    # abandoned numpy for sum: had too many indices
    tmp = sum([term(k) for k in range(j)])
    # sum = np.sum(np.tensordot(Z(k), U[:,j-k-1], axes=(-1,0)))
    # Z(0) U_j = b
    U[:,j], exitCode = scipy.sparse.linalg.gmres(A, -V(j) - tmp)
    assert exitCode == 0

# def U_fn(rho, t):
#     discrete_f = f(np.arange(1, N_S + 1))
#     discrete_T = T_basis(np.arange(1, N_T + 1))
#     return np.einsum("ni,n,i", U, discrete_f(rho), discrete_T(t))


u = dt*np.fft.rfft(U, axis=1) # u[n,i] = u(rho_n, omega_i)
omega = 2*np.pi * np.fft.rfftfreq(U.shape[1], dt) # rad/s
j = u / 1j / omega.reshape((1, -1)) / mu
plt.figure()
plt.plot(np.arange(0, N_T*dt, dt), U[0,:])
plt.plot(np.arange(0, N_T*dt, dt), U[N_S//2,:])
plt.xlabel("t (s)")
plt.title("j at phi=0")

A = np.exp(-1j * omega * t_0 - (T * omega / 8 / c)**2) / c
plt.figure()
plt.plot(omega/c, np.abs(A))
plt.title("spectrum excitation")
plt.xlabel("$\\omega$/c (m$^{-1}$)")
plt.ylabel("A (s/m)")
j_0 = np.abs(j / A.reshape((1, -1)))
fig, axes = plt.subplots(2, 2, sharex='col')
# FROM OMEGA[10] INSTABILITY STARTS TO DEVELOP AND ONLY BECOMES WORSE: TODO
# instability due to divide by A: becomes nearly zero
axes[0,0].plot(omega[:], j_0[0,:], label=f"{curve_points[[_x, _y],0]} m")
axes[0,0].plot(omega[:], j_0[N_S//2,:], label=f"{curve_points[[_x, _y],N_S//2]} m")
axes[0,0].set_title("normalized current")
# axes[0,0].set_xlabel("$\\omega$ (rad/s)")
axes[0,0].set_ylabel("j$_0$")
axes[0,0].set_ylim(0, .03)
axes[0,0].legend()

axes[0,1].plot(np.arctan2(curve_points[_y,:-1], curve_points[_x,:-1]), j_0[:,1], label=f"$\\omega$={omega[1]} rad/s")
axes[0,1].plot(np.arctan2(curve_points[_y,:-1], curve_points[_x,:-1]), j_0[:,2], label=f"$\\omega$={omega[2]} rad/s")
axes[0,1].plot(np.arctan2(curve_points[_y,:-1], curve_points[_x,:-1]), j_0[:,3], label=f"$\\omega$={omega[3]} rad/s")
axes[0,1].set_title("normalized current")
# axes[0,1].set_xlabel("$\\phi$ (rad)")
axes[0,1].set_ylabel("j$_0$")
axes[0,1].legend()

# Analytical solution
# This is for incoming field e^jkx

# def e_z(rho):
#     phi = np.atan2(rho[_y], rho[_x])  # rad
#     rho = np.linalg.norm(rho, axis=0) # m
#     a = R                             # m
#     k = 1                             # 1/m TODO

#     n = np.arange(np.ceil(np.min([k*rho, k*a], axis=0)) + 2)
#     return np.sum(1j**n * (fns.jn(n, k*rho) - fns.jn(k*a)/fns.hankel2(n, k*a)*fns.hankel2(k*rho)) * np.exp(1j * n * phi))

# d e_z / d rho = sum j^n (k J_n'(k rho) - k J_n(k a) / H_n^(2)(k a) * H_n^(2)'(k rho)) e^(j n phi)
# at a: sum j^n k (J_n'(k a) H_n^(2)(k a) - J_n(k a) * H_n^(2)'(k a)) / H_n^(2)(k a) * e^(j n phi)
# using Wronskian: sum 2 j j^n k e^(j n phi) / pi k a H_n^(2)(k a) 

def j_z(phi):
    # omega = 1                   # rad TODO
    k = omega.reshape(-1, 1, 1)/c # rad/m TODO
    a = R                         # m

    n = np.arange(np.ceil(np.max(k)*a) + 2).reshape(1, -1, 1)
    return 1/1j/omega.reshape((-1, 1))/mu * 2 * np.sum(1j**(n+1) * k * np.exp(1j*n*phi) / np.pi / k / a / fns.hankel2(n, k*a), axis=1)
phi = np.linspace(-np.pi, np.pi, 128)
j_z = np.abs(j_z(phi))
axes[1,1].plot(phi, j_z[1], label=f"{omega[1]} rad/s")
axes[1,1].plot(phi, j_z[2], label=f"{omega[2]} rad/s")
axes[1,1].plot(phi, j_z[3], label=f"{omega[3]} rad/s")
axes[1,1].set_title("analytical current")
axes[1,1].set_xlabel("$\\phi$ (rad)")
axes[1,1].set_ylabel("j$_z$")
axes[1,1].legend()

axes[1,0].plot(omega, j_z[:,0], label=f"{curve_points[[_x, _y],0]} m")
axes[1,0].plot(omega, j_z[:,N_S//2], label=f"{curve_points[[_x, _y],N_S//2]} m")
axes[1,0].set_title("analytical current")
axes[1,0].set_xlabel("$\\omega$ (rad/s)")
axes[1,0].set_ylabel("j$_z$")
axes[1,0].legend()

plt.show()

def fprime1(t):
    """
    Time derivative of the source pulse f(t), stable for narrow pulses.
    """
    alpha = 4/T1
    norm = 4 / (T1 * np.sqrt(np.pi))
    x = alpha * (t - t_01)
    mask = np.abs(x) < 20  # compute only significant values
    result = np.zeros_like(t, dtype=float)
    result[mask] = -2 * alpha**2 * (t[mask] - t_01) * norm * np.exp(-(x[mask])**2)
    return result

def incident_field_matrix1(rho, t, NG=12):
    """
    Compute Ei(rho, t) for each element using Gauss-Legendre quadrature.
    rho and t: 2D arrays of shape (N_rho, N_t)
    Returns: Ei, same shape
    """
    N_rho, N_t = rho.shape
    Ei = np.zeros_like(rho) #(N_rho,N_t)

    # Gauss-Legendre nodes and weights on [-1,1], scaled for [0,1]
    xi, wi = np.polynomial.legendre.leggauss(NG)
    wi = wi * (-mu / (4*np.pi)) / 2  # scale for [0,1]

    for i in range(N_rho):
        for j in range(N_t):
            r = rho[i,j]
            tt = t[i,j]

            # Causality: only compute for t > r/c
            if tt <= r / c:
                Ei[i,j] = 0.0
                continue

            # Compute u_max, avoid NaN
            arg = c / r * (tt - t_01)
            arg = np.maximum(arg, 1.0)
            umax = np.arccosh(arg)

            # Gauss nodes in u
            u = (umax / 2) * (1 + xi)  # shape (NG,)

            # Integrand
            arg_t = tt - (r / c) * np.cosh(u)
            integrand = fprime1(arg_t)

            # Weighted sum
            Ei[i,j] = np.sum(wi * integrand) * umax

    return Ei
timeframe = t_01 + np.arange(N_T) * dt
eps = R/100000
radius = np.linspace(eps,R,N_S)

Meshradius,Meshtime = np.meshgrid(radius,timeframe, indexing='ij')

Ei = incident_field_matrix1(Meshradius,Meshtime)
def animate_Ei_line(radius, Ei, dt, save_path=None):
    """
    Animate Ei(r, t) as a line plot over radius.
    radius: 1D array (N_r)
    Ei: 2D array (N_r, N_t)
    dt: time step
    save_path: optional filename to save mp4 or gif
    """
    fig, ax = plt.subplots(figsize=(6,4))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(radius[0], radius[-1])
    # auto scale based on Ei
    ax.set_ylim(np.min(Ei), np.max(Ei)/2)
    ax.set_xlabel("Radius (m)")
    ax.set_ylabel("Ei(r, t)")
    
    N_t = Ei.shape[1]

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        line.set_data(radius, Ei[:, frame])
        ax.set_title(f"Time = {frame*dt*1e9:.2f} ns")
        return (line,)

    anim = FuncAnimation(fig, update, frames=N_t, init_func=init,
                         blit=False, interval=50)

    if save_path is not None:
        anim.save(save_path, fps=20, dpi=150)

    plt.show()
    return anim

anim = animate_Ei_line(radius, Ei, dt)