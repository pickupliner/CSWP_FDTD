import numpy as np
import scipy
import scipy.special as fns
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import functools
from matplotlib.animation import FuncAnimation


class MOT:
    # =====================
    # Initialization
    # =====================
    def __init__(self,
                 R=1.0,
                 N_S=32,
                 N_T=512,
                 N_G=8,
                 c=3e8,
                 mu=np.pi * 4e-7):

        # Physical constants
        self.c = c
        self.mu = mu
        self.R = R

        # Discretization
        self.N_S = N_S
        self.N_T = N_T
        self.N_G = N_G
        self.dt = R / c / 10

        # Indices
        self._x, self._y = 0, 1

        # Geometry
        self._build_geometry()

        # Time pulse parameters
        self._init_incident_pulse()

        # Quadrature
        self._init_quadrature()

        # Operator cache
        self._Z_cache = {}

        # Solution container
        self.U = np.zeros((self.N_S, self.N_T))

    # =====================
    # Geometry
    # =====================
    def curve(self, s):#m
        arg = 2 * np.pi * s
        return self.R * np.array([np.cos(arg), np.sin(arg)])

    def _build_geometry(self):#m
        s = np.linspace(0, 1, self.N_S + 1)
        self.curve_points = self.curve(s)

        self.tangents = self.curve_points[:, 1:] - self.curve_points[:, :-1]
        self.l = np.linalg.norm(self.tangents, axis=0)

        self.L = np.minimum(self.l, 2 * self.c * self.dt)

    def rho_n(self, s): #m
        shape = (2, self.N_S) + (1,) * len(s.shape)
        return (self.curve_points[:, :-1].reshape(shape)
                + s * self.tangents.reshape(shape))
    
    def create_spacetimemesh(self):
        self.timeframe = self.t_01 + np.arange(self.N_T)* self.dt
        eps=self.R/10000
        self.radius  = np.linspace(eps,self.R,self.N_S)

        self.MeshR,self.Mesht = np.meshgrid(self.radius,self.timeframe,indexing= 'ij')
        return None

    # =====================
    # Incident field
    # =====================
    def _init_incident_pulse(self):
        self.t_0 = self.N_T / np.log2(self.N_T / 8) * self.dt # s
        self.T = self.c * self.t_0 / np.sqrt(2 * np.pi) / 2 #m
        self.T1 = 20*self.dt
        self.t_01 = 10*self.T1

    def E_i1(self, rho, t):
        gamma = 4 / self.T * (self.c * (t - self.t_0) - rho[self._x])
        return 4 / self.T / np.sqrt(np.pi) * np.exp(-gamma**2)

    # =====================
    # Quadrature
    # =====================
    def _init_quadrature(self):
        s, w = np.polynomial.legendre.leggauss(self.N_G)
        self.s = (s + 1) / 2
        self.w = w / 2

        self.s = self.s.reshape((1, 1, self.N_G))
        self.w = self.w.reshape((1, 1, self.N_G))

    def F(self, k, rho_m, rho_p):
        dist = np.linalg.norm(
            rho_m.reshape(rho_m.shape + (1, 1, 1)) - rho_p,
            axis=0
        ) / self.c

        a = np.max([np.broadcast_to([k*self.dt], dist.shape), dist], axis=0)
        b = np.max([np.broadcast_to([(k+1)*self.dt], dist.shape), dist], axis=0)

        return np.log((b + np.sqrt(b**2 - dist**2)) /
                      (a + np.sqrt(a**2 - dist**2)))

    # =====================
    # Operators
    # =====================
    def Z(self, k):
        if k in self._Z_cache:
            return self._Z_cache[k]

        tmp = np.sqrt((2 * self.c * self.dt)**2 - self.L**2)
        Z0 = np.diag(
            -self.L / (2 * np.pi) * np.log((2 * self.c * self.dt + tmp) / self.L)
            - self.c * self.dt / np.pi * np.arctan(self.L / tmp)
        )

        m = np.arange(self.N_S).reshape(self.N_S, 1)
        n = np.arange(self.N_S).reshape(1, self.N_S)

        quad = np.sum(
            self.w * self.F(k,
                            self.curve_points[:, m],
                            self.rho_n(self.s)[:, n]),
            axis=-1
        ).reshape(self.N_S, self.N_S)

        Z = -self.l[n] / (2 * np.pi) * quad
        Z = np.where((k == 0) & (m == n), Z0, Z)

        self._Z_cache[k] = Z
        return Z

    def V(self, j,E_i):
        return E_i(self.curve_points[:, :-1], j * self.dt)

    # =====================
    # Time-marching solver
    # =====================
    def solve(self,E_i):
        A = self.Z(0)
        b = -self.V(0,E_i) 
        self.U[:, 0], info = spla.gmres(A, b)
        assert info == 0 # error if equation could not be solved

        for j in range(1, self.N_T):
            # k = np.arange(0, j-1 + 1) # k = 0, .., j-1
            rhs = -self.V(j,E_i)
            conv = sum(self.Z(k) @ self.U[:, j - k - 1]
                       for k in range(j))
            self.U[:, j], info = spla.gmres(A, rhs - conv)
            assert info == 0

        return self.U
    
    #======================
    #6.2.
    #======================
    def fprime(self,t):
        """
        Time derivative of the source pulse f(t), stable for narrow pulses.
        """
        alpha = 4/self.T1
        norm = 4 / (self.T1 * np.sqrt(np.pi))
        x = alpha * (t - self.t_01)
        mask = np.abs(x) < 20  # compute only significant values
        result = np.zeros_like(t, dtype=float)
        result[mask] = -2 * alpha**2 * (t[mask] - self.t_01) * norm * np.exp(-(x[mask])**2)
        return result
    

    def E_i2(self,rho,t):
        """
        Compute Ei(rho, t) for each element using Gauss-Legendre quadrature.
        rho and t: 2D arrays of shape (N_rho, N_t)
        Returns: Ei, same shape
        """
        N_rho, N_t = rho.shape
        Ei = np.zeros_like(rho) #(N_rho,N_t)

        # Gauss-Legendre nodes and weights on [-1,1], scaled for [0,1]
        xi, wi = np.polynomial.legendre.leggauss(self.N_G)
        wi = wi * (-self.mu / (4*np.pi)) / 2  # scale for [0,1]

        for i in range(N_rho):
            for j in range(N_t):
                r = rho[i,j]
                tt = t[i,j]

                # Causality: only compute for t > r/c
                if tt <= r / self.c:
                    Ei[i,j] = 0.0
                    continue

                # Compute u_max, avoid NaN
                arg = self.c / r * (tt - self.t_01)
                arg = np.maximum(arg, 1.0)
                umax = np.arccosh(arg)

                # Gauss nodes in u
                u = (umax / 2) * (1 + xi)  # shape (NG,)

                # Integrand
                arg_t = tt - (r / self.c) * np.cosh(u)
                integrand = self.fprime(arg_t)

                # Weighted sum
                Ei[i,j] = np.sum(wi * integrand) * umax

        return Ei
    # =====================
    # Post-processing
    # =====================
    def spectrum(self):
        u = self.dt * np.fft.rfft(self.U, axis=1)
        omega = 2 * np.pi * np.fft.rfftfreq(self.U.shape[1], self.dt)
        j = u / (1j * omega.reshape((1,-1))) / self.mu
        return omega, j

    # =====================
    # Visualization
    # =====================
    def plot_geometry(self):
        plt.plot(*self.curve_points)
        plt.axis("equal")
        plt.title("Geometry")
        plt.show()

    def plot61(self):
        omega, j = mot.spectrum()
        plt.figure()
        plt.plot(np.arange(0, self.N_T*self.dt, self.dt), self.U[0,:])
        plt.plot(np.arange(0, self.N_T*self.dt, self.dt), self.U[self.N_S//2,:])
        plt.xlabel("t (s)")
        plt.title("j at phi=0")

        A = np.exp(-1j * omega * self.t_0 - (self.T * omega / 8 / self.c)**2) / self.c
        plt.figure()
        plt.plot(omega/self.c, np.abs(A))
        plt.title("spectrum excitation")
        plt.xlabel("$\\omega$/c (m$^{-1}$)")
        plt.ylabel("A (s/m)")
        j_0 = np.abs(j / A.reshape((1, -1)))
        fig, axes = plt.subplots(2, 2, sharex='col')
        # FROM OMEGA[10] INSTABILITY STARTS TO DEVELOP AND ONLY BECOMES WORSE: TODO
        # instability due to divide by A: becomes nearly zero
        axes[0,0].plot(omega[:], j_0[0,:], label=f"{self.curve_points[[self._x, self._y],0]} m")
        axes[0,0].plot(omega[:], j_0[self.N_S//2,:], label=f"{self.curve_points[[self._x, self._y],self.N_S//2]} m")
        axes[0,0].set_title("normalized current")
        # axes[0,0].set_xlabel("$\\omega$ (rad/s)")
        axes[0,0].set_ylabel("j$_0$")
        axes[0,0].set_ylim(0, .03)
        axes[0,0].legend()

        axes[0,1].plot(np.arctan2(self.curve_points[self._y,:-1], self.curve_points[self._x,:-1]), j_0[:,1], label=f"$\\omega$={omega[1]} rad/s")
        axes[0,1].plot(np.arctan2(self.curve_points[self._y,:-1], self.curve_points[self._x,:-1]), j_0[:,2], label=f"$\\omega$={omega[2]} rad/s")
        axes[0,1].plot(np.arctan2(self.curve_points[self._y,:-1], self.curve_points[self._x,:-1]), j_0[:,3], label=f"$\\omega$={omega[3]} rad/s")
        axes[0,1].set_title("normalized current")
        # axes[0,1].set_xlabel("$\\phi$ (rad)")
        axes[0,1].set_ylabel("j$_0$")
        axes[0,1].legend()
        def j_z(phi):
            # omega = 1                   # rad TODO
            k = omega.reshape(-1, 1, 1)/self.c # rad/m TODO
            a = self.R                         # m

            n = np.arange(np.ceil(np.max(k)*a) + 2).reshape(1, -1, 1)
            return 1/1j/omega.reshape((-1, 1))/self.mu * 2 * np.sum(1j**(n+1) * k * np.exp(1j*n*phi) / np.pi / k / a / fns.hankel2(n, k*a), axis=1)
        phi = np.linspace(-np.pi, np.pi, 128)
        j_z = np.abs(j_z(phi))
        axes[1,1].plot(phi, j_z[1], label=f"{omega[1]} rad/s")
        axes[1,1].plot(phi, j_z[2], label=f"{omega[2]} rad/s")
        axes[1,1].plot(phi, j_z[3], label=f"{omega[3]} rad/s")
        axes[1,1].set_title("analytical current")
        axes[1,1].set_xlabel("$\\phi$ (rad)")
        axes[1,1].set_ylabel("j$_z$")
        axes[1,1].legend()

        axes[1,0].plot(omega, j_z[:,0], label=f"{self.curve_points[[self._x,self._y],0]} m")
        axes[1,0].plot(omega, j_z[:,self.N_S//2], label=f"{self.curve_points[[self._x, self._y],self.N_S//2]} m")
        axes[1,0].set_title("analytical current")
        axes[1,0].set_xlabel("$\\omega$ (rad/s)")
        axes[1,0].set_ylabel("j$_z$")
        axes[1,0].legend()

        plt.show()
        
        return None
    
    
    def animate_E_i(self, radius, Ei):
        fig, ax = plt.subplots()
        line, = ax.plot([], [])
        ax.set_xlim(radius[0], radius[-1])
        ax.set_ylim(Ei.min(), Ei.max()/2)

        def update(i):
            line.set_data(radius, Ei[:, i])
            ax.set_title(f"t = {i * self.dt:.2e} s")
            return line,

        anim = FuncAnimation(fig, update, frames=Ei.shape[1])
        plt.show()
        return anim
    



mot = MOT(N_S=100, N_T=512)
Ei1= mot.E_i1(mot.curve_points[:, :-1],mot.dt)
mot.create_spacetimemesh()

Ei2=mot.E_i2(mot.MeshR,mot.Mesht)

print(np.size(Ei1))
print(np.size(Ei2))

#radius= mot.radius
#mot.animate_E_i(radius,Ei1)
#U = mot.solve()
#omega, mot.j = mot.spectrum()


#mot.animate_Ei(radius,Ei2)