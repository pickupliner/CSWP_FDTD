import numpy as np
import scipy
import scipy.special as fns

# Not finished yet (validation)
# Constants (mu, N_S, ..) are placeholders
# Indices are a bit of a mess, my apologies

# indices to get x- and y-coordinate
_x, _y = 0, 1
# speed of wave
c = 3e8 # m/s
# permeability TODO
mu = 1

# spatial points
N_S = 17
# temporal 
N_T = 11
dt = 0.3 # s
# Gauss quad order
N_G = 7

# curve

R = 1 # m
def curve(s):
    # curve(0) = curve(1)
    # s in range [0, 1]
    arg = 2*np.pi * s                             # rad
    return R*np.array([np.cos(arg), np.sin(arg)]) # m  (2, s.shape)
# (the first and last point are equal for simplicity)
curve_points = curve(np.linspace(0, 1, N_S + 1))          # m  (2, N_S + 1)
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

# incident field
def E_i(rho, t):
    t_0 = 3 # s
    T = 1 # m
    gamma = 4/T * (c*(t - t_0) - rho[_x])         # 1
    return 4/T/np.sqrt(np.pi) * np.exp(-gamma**2) # 1/m

# basis functions

# T(i)(t) = 1 if (i-1) dt < t < i dt
def T(i):
    def T(t):
        return np.where((-dt < t) & (t < 0), 1, 0)
    return lambda t: T(t - i*dt)

# f(n)(rho) = 1 if rho is on segment n
def f(n):
    def f(rho):
        # if d_1 + d_2 = d_0 then rho is on the line
        d_0 = l[n]
        d_1 = np.linalg.norm(rho - curve_points[:,n], axis=0)
        d_2 = np.linalg.norm(curve_points[:,n+1] - rho, axis=0)
        return np.isclose(d_1 + d_2, d_0)
    return f

def F(k, rho_m, rho_prime):
    tmp = np.linalg.norm(rho_m.reshape(rho_m.shape + (1,1,1,)) - rho_prime, axis=0)/c
    a = np.max([np.broadcast_to([k*dt], tmp.shape), tmp], axis=0)
    b = np.max([np.broadcast_to([(k+1)*dt], tmp.shape), tmp], axis=0)
    return np.log((b + np.sqrt(b**2 - tmp**2))/(a + np.sqrt(a**2 - tmp**2)))

# s: -1 to 1
s, w = np.polynomial.legendre.leggauss(N_G)
# s: 0 to 1
s, w = (s.reshape((1, 1, N_G)) + 1)/2, w.reshape((1, 1, N_G))/2

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
    # print(k.shape)
    # print(Z(k).shape)
    # print(U[:,j-k-1].shape)
    def term(k):
        return np.tensordot(Z(k), U[:,j-k-1], axes=(-1,0))
    # abandoned numpy for sum: had too many indices
    tmp = sum([term(k) for k in range(j)])
    # sum = np.sum(np.tensordot(Z(k), U[:,j-k-1], axes=(-1,0)))
    # Z(0) U_j = b
    U[:,j], exitCode = scipy.sparse.linalg.gmres(A, -V(j) - tmp)
    assert exitCode == 0

def U_fn(rho, t):
    discrete_f = f(np.arange(1, N_S + 1))
    discrete_T = T(np.arange(1, N_T + 1))
    return np.einsum("ni,n,i", U, discrete_f(rho), discrete_T(t))

# Analytical solution
# This is for incoming field e^jkx
# TODO solution for assigned field

def e_z(rho):
    phi = np.atan2(rho[_y], rho[_x])  # rad
    rho = np.linalg.norm(rho, axis=0) # m
    a = R                             # m
    k = 1                             # 1/m TODO

    n = np.arange()
    return np.sum(1j**n * (fns.jn(n, k*rho) - fns.jn(k*a)/fns.hankel2(n, k*a)*fns.hankel2(k*rho)) * np.exp(1j * n * phi))

# d e_z / d rho = sum j^n (k J_n'(k rho) - k J_n(k a) / H_n^(2)(k a) * H_n^(2)'(k rho)) e^(j n phi)
# at a: sum j^n k (J_n'(k a) H_n^(2)(k a) - J_n(k a) * H_n^(2)'(k a)) / H_n^(2)(k a) * e^(j n phi)
# using Wronskian: 2 j j^n k e^(j n phi) / pi k a H_n^(2)(k a) 

def j_z(phi):
    omega = 1 # rad TODO
    k = 1     # TODO
    a = R     # m

    n = np.arange()
    return 2 * 1j**(n+1) * k * np.exp(1j*n*phi) / np.pi / k / a / fns.hankel2(n, k*a)
