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
                 curve,          # [m] (2, N_S+1) points on PEC
                #  R=1.0,          # [m]   Radius Pec
                #  N_S=42,         # [1]   spatial nodes
                 dt=1e-8,
                 N_T=512,        # [1]   temporal nodes
                 N_G=8,          # [1]   amount of weights for quadrature
                 c=3e8,          # [m/s] light constant
                 mu=np.pi * 4e-7 # [H/m] permeability
                 ):

        # Physical constants
        self.c = c
        self.mu = mu

        # Discretization
        self.N_S = curve.shape[1] - 1
        self.N_T = N_T
        self.N_G = N_G
        
        self.dt = dt

        # Indices
        self._x, self._y = 0, 1  #whenever this is used first index is x-coordinates and second is y coordinates

        # Quadrature
        self._init_quadrature() # creates quadrature

        # incident field
        self._init_incident_pulse()

        # Geometry
        self.curve = curve
        self._build_geometry() # creates the geometry of the boundary which is a (2,N_S) matrix with 0 x and 1 y

        self._init_F() # creates F array for use in solver
        self._init_Z() # creates Z array for use in solver

        # Solution container
        self.U = np.zeros((self.N_S, self.N_T))  # solution is stored here
         
    # =====================
    # Geometry
    # =====================

    def _build_geometry(self):
        """
        _build_geometry creates the tangent lines and the coordinates of the circle 
        this is all on the boundary.
        Assumes quadrature allready initiated.
        
        :param self: class objext
        """

        # [m] (2, N_S) centers of all the segments
        self.rho = (self.curve[:, 1:] + self.curve[:, :-1])/2
        
        # [m] (2, N_S) these are all the tangent vectors
        self.tangents = self.curve[:, 1:] - self.curve[:, :-1]

        # [m] (N_S,) this gets all the norms of the tangents/segments
        self.l = np.linalg.norm(self.tangents, axis=0)
        
        self.L = np.minimum(self.l, 2 * self.c * self.dt)
        # we check because under the sqrt it shouldnt be 0

        # linear interpolation for gaussian quadrature
        # [m] (xy, N_S, N_G) g coordinates along m segments
        self.rhop = self.curve[:, :-1].reshape((2,self.N_S,1)) \
                  + self.coordquad*self.tangents[:, :].reshape((2,self.N_S,1))

    def create_spacetimemesh_6_2(self,R,discretizationpointsrho,discretizationpointstime):
        """
        
        creates mesh in spacetime You can return with recast such that 
        (2,N_s,m,n,N_rho,N_T,N_G)
        :param self: Description
        
        """
        N_rho = discretizationpointsrho
        N_time = discretizationpointstime

        self.timeframe = self.t_01 + np.arange(self.N_T)* self.dt
        eps=R/10000
        self.radius  = np.linspace(eps,R,N_rho)

        self.MeshR,self.Mesht = np.meshgrid(self.radius,self.timeframe,indexing= 'ij')
        
        return None

    # =====================
    # Incident field
    # =====================
    def _init_incident_pulse(self):
        self.T1 = 20*self.dt
        self.t_01 = 10*self.T1

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
        wi = wi * (-self.mu / (4*np.pi))  # scale for [0,1]

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
    # Quadrature
    # =====================
    def _init_quadrature(self):
        coordoriginal, woriginal = np.polynomial.legendre.leggauss(self.N_G)
        self.coordquad = (coordoriginal + 1) / 2
        self.weights = woriginal / 2

        self.coordquad = self.coordquad
        self.weights = self.weights

    def _init_F(self):
        """
        Inits self.F,
        assumes geometry is already initiated.
        
        :param self: Description
        """

        #                              xy m  n         g
        rho_resh  = self.rho .reshape((2,-1, 1,        1       ))
        rhop_resh = self.rhop.reshape((2, 1, self.N_S, self.N_G))
        # dist_c : (m,n,g)
        dist_c = (np.linalg.vector_norm(rho_resh - rhop_resh, axis=0)/self.c)
        #                                 k  m  n  g
        k = np.arange(self.N_T).reshape((-1, 1, 1, 1))
        # a,b : (k,m,n,g)
        a = np.maximum(k*self.dt, dist_c)
        b = np.maximum((k+1)*self.dt, dist_c)
        # F : (k,m,n,g)
        self.F = np.log((b + np.sqrt(b**2 - dist_c**2))/(a + np.sqrt(a**2 - dist_c**2)))

    def _init_Z(self):
        """
        Inits self.Z,
        assumes quadrature, geometry and F are already initiated.
        
        :param self: Description
        """

        n = np.arange(self.N_S).reshape((1,1,-1))
        # sum with g from 0 to N_G over w_g . F_kmng
        quad = np.einsum('g,kmng->kmn', self.weights, self.F)
        # Z : (k,m,n)
        self.Z = -self.l[n]/2/np.pi * quad

        # diagonal of Z_0
        for m in range(self.N_S):
            root = np.sqrt((2*self.c*self.dt)**2 - self.L[m]**2)
            self.Z[0,m,m] = -self.L[m]/2/np.pi * np.log((2*self.c*self.dt + root)/self.L[m]) \
                           - self.c*self.dt/np.pi * np.arctan(self.L[m]/root)

    def V1(self, j, E_i):
        return E_i(self.rho[:, :], j * self.dt)

    # =====================
    # Time-marching solver
    # =====================
    def solve(self, V):
        """
        Solves the MOT problem with the excitation V.
        Assumes Z and all the rest are initiated.
        
        :param self: Description
        :param V: a 2D-array of shape (N_T, N_S). V_jm: the incident field at time j dt and at position rho_m
        """
        assert V.shape == (self.N_T, self.N_S)

        # U : (n,i)
        # solve Z_0 . U = -V_0
        self.U[:,0], info = spla.gmres(A=self.Z[0,:,:], b=-V[0,:])
        # throw error if problem could not be solved
        assert info == 0 

        for j in range(1, self.N_T):
            k = np.arange(j)
            i = j - k
            # sum with k from 0 to j-1 over Z_k . U_(j-k)
            conv = np.einsum(f'kmn,mk->n', self.Z[k,:,:], self.U[:,i])
            # solve Z_0 . U = -V_j - sum_k Z_k . U_(j-k)
            self.U[:,j], info = spla.gmres(
                A=self.Z[0,:,:], 
                b= -V[j,:] - conv
            )
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
    
    def analyticalzeros(self,totalnorder,amountofzeros):
        zeros = []
        for n in range(totalnorder+1):
            zeros.append((self.c/self.R)*fns.jn_zeros(n,amountofzeros))

        self.zeros = zeros
        return zeros


    # =====================
    # Post-processing
    # =====================
    def positivespectrum(self):
        # factor in front is to account for the discrepency between the continuous and discrete FT (see https://stackoverflow.com/questions/24077913/discretized-continuous-fourier-transform-with-numpy)
        u = self.dt / np.sqrt(2*np.pi) * np.fft.rfft(self.U, axis=1)
        self.omega = 2 * np.pi * np.fft.rfftfreq(self.U.shape[1], self.dt)

        self.j = np.zeros_like(u, dtype=complex)
        self.j[:, 1:] = u[:, 1:] / (1j * self.omega[1:].reshape((1,-1))) / self.mu

        return self.omega, self.j


    # # =====================
    # # analytical
    # # =====================
    # def j_z(self,phi,omega):
    #     k = omega.reshape(-1, 1, 1)/self.c # rad/m
        
    #     a = self.R                         # m

    #     n = np.arange(np.ceil(np.max(k)*a) + 2).reshape(1, -1, 1)
    #     return 1/1j/omega.reshape((-1, 1))/self.mu * 2 * np.sum(1j**(n+1) * k * np.exp(1j*n*phi) / np.pi / k / a / fns.hankel2(n, k*a), axis=1)
    

    # def analytical6_1_(self, R):

    #     phi = np.linspace(-np.pi, np.pi, 128)
    #     self.jzanalyticalfrequency = self.j_z(phi,self.omega)
    #     Jz = np.abs(self.j_z(phi,self.omega))
    #     self.jzanalyticaltime = np.fft.irfft(self.jzanalyticalfrequency,axis=1)

    #     return phi,Jz
        
    # =====================
    # Visualization
    # =====================
    def plot_geometry(self):
        plt.plot(*self.rho, 'o')
        plt.plot(*self.rhop[:,0,:], '.')
        plt.axis("equal")
        plt.title("Geometry")
        plt.show()
    
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
    # =====================
    # Plot current on circle
    # =====================
    def plot_current_on_circle(self, time_index, mode="polar"):
        """
        Plot surface current on the circular boundary.

        Parameters
        ----------
        time_index : int
            Time step index
        mode : str
            "angle"  -> j vs phi
            "polar"  -> polar plot
            "vector" -> quiver plot on circle
        """

        # Element midpoints
        rho_mid = 0.5 * (self.curve[:, :-1] + self.curve[:, 1:])

        # Angle of each element
        phi = np.arctan2(rho_mid[self._y], rho_mid[self._x])

        # Sort by angle for clean plotting
        idx = np.argsort(phi)
        phi = phi[idx]

        # Current at this time step
        j = self.U[:, time_index][idx]

        # ----------------------------------
        # 1) j vs angle
        # ----------------------------------
        if mode == "angle":
            plt.figure()
            plt.plot(phi, j)
            plt.xlabel(r"$\phi$ (rad)")
            plt.ylabel("Surface current")
            plt.title(f"Surface current at t = {time_index*self.dt:.2e} s")
            plt.grid(True)
            plt.show()

        # ----------------------------------
        # 2) Polar plot
        # ----------------------------------
        elif mode == "polar":
            plt.figure()
            ax = plt.subplot(111, projection="polar")
            ax.plot(phi, np.abs(j))
            ax.set_title(f"|Surface current| at t = {time_index*self.dt:.2e} s")
            plt.show()

        # ----------------------------------
        # 3) Vector plot on the circle
        # ----------------------------------
        elif mode == "vector":
            x = rho_mid[self._x, idx]
            y = rho_mid[self._y, idx]

            # Tangent direction (current flows tangentially)
            tangent = self.tangents[:, idx]
            tangent /= np.linalg.norm(tangent, axis=0)

            jx = j * tangent[self._x]
            jy = j * tangent[self._y]

            plt.figure()
            plt.quiver(x, y, jx, jy, scale=1, scale_units="xy")
            plt.gca().set_aspect("equal")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"Surface current vectors at t = {time_index*self.dt:.2e} s")
            plt.show()

        else:
            raise ValueError("mode must be 'angle', 'polar', or 'vector'")
    # =====================
    # Animate surface current on circle (FIXED)
    # =====================
    def animate_current_on_circle1(self, R, scale=1.0, interval=40):
        """
        Animate the surface current on the circular boundary.
        """

        # Element midpoints
        rho_mid = 0.5 * (self.curve[:, :-1] + self.curve[:, 1:])
        phi = np.arctan2(rho_mid[self._y], rho_mid[self._x])

        # Sort by angle
        idx = np.argsort(phi)
        phi = phi[idx]

        # Normalize current
        max_j = np.max(np.abs(self.U))
        if max_j == 0:
            max_j = 1.0

        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.set_xlim(-1.5 * R, 1.5 * R)
        ax.set_ylim(-1.5 * R, 1.5 * R)

        line, = ax.plot([], [], lw=2)

        # IMPORTANT: keep reference
        self._ani = None

        def init():
            x = R * np.cos(phi)
            y = R * np.sin(phi)
            line.set_data(x, y)
            return line,

        def update(frame):
            j = self.U[:, frame][idx]
            r = R * (1 + scale * j / max_j)

            x = r * np.cos(phi)
            y = r * np.sin(phi)

            line.set_data(x, y)
            ax.set_title(f"Surface current, t = {frame * self.dt:.2e} s")
            return line,

        self._ani = FuncAnimation(
            fig,
            update,
            frames=self.N_T,
            init_func=init,
            interval=interval,
            blit=False  # <<<
        )

        plt.show()
        return self._ani

    # =====================
    # Animate 1D surface current vs angle
    # =====================
    def animate_current_1d(self, interval=30):
        """
        Animate the surface current along the circle in 1D
        (current magnitude vs angular position)
        """
        # Element midpoints
        rho_mid = 0.5 * (self.curve[:, :-1] + self.curve[:, 1:])
        phi = np.arctan2(rho_mid[self._y], rho_mid[self._x])

        # Sort by angle
        idx = np.argsort(phi)
        phi = phi[idx]

        # Normalize current for visualization
        max_j = np.max(np.abs(self.U))
        if max_j == 0:
            max_j = 1.0

        fig, ax = plt.subplots()
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-1.1 * max_j, 1.1 * max_j)
        ax.set_xlabel(r"$\phi$ (rad)")
        ax.set_ylabel("Surface current")
        ax.set_title("Surface current along the circle")

        line, = ax.plot([], [], lw=2)
        self._ani1d = None  # keep reference

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            j = self.U[:, frame][idx]
            line.set_data(phi, j)
            ax.set_title(f"Surface current along circle, t = {frame*self.dt:.2e}s")
            return line,

        self._ani1d = FuncAnimation(
            fig,
            update,
            frames=self.N_T,
            init_func=init,
            interval=interval,
            blit=False
        )

        plt.show()
        return self._ani1d
    #============
    #animating the incoming E-field
    #============
    def animate_Ei1(self, R, E_i, xlim=(-2, 2), interval=30):
        x = np.linspace(xlim[0]*R, xlim[1]*R, 400)
        rho = np.zeros((2, x.size))
        rho[0, :] = x

        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(x.min(), x.max())
        # ax.set_ylim(0, 1.2 * 4 / self.T / np.sqrt(np.pi))
        ax.set_xlabel("x (m)")
        ax.set_ylabel(r"$E_i$")
        ax.set_title("Incoming electric pulse")

        def update(frame):
            t = frame * self.dt
            Ei = E_i(rho, t)
            line.set_data(x, Ei)
            ax.set_title(f"Incoming electric pulse, t = {t:.2e} s")
            return line,

        ani = FuncAnimation(
            fig,
            update,
            frames=self.N_T,
            interval=interval,
            blit=False
        )

        plt.show()
        return ani
    def animate_Ei1_2D(self, R, E_i, interval=30):
        # Spatial grid
        x = np.linspace(-2*R, 2*R, 300)
        y = np.linspace(-2*R, 2*R, 300)
        X, Y = np.meshgrid(x, y, indexing="ij")

        rho = np.zeros((2, X.size))
        rho[0, :] = X.ravel()
        rho[1, :] = Y.ravel()

        # Figure
        fig, ax = plt.subplots()
        im = ax.imshow(
            np.zeros_like(X),
            extent=[y.min(), y.max(), x.min(), x.max()],
            origin="lower",
            cmap="RdBu",
            vmin=-1,
            vmax=1,
            animated=True
        )

        ax.set_xlabel("y (m)")
        ax.set_ylabel("x (m)")
        ax.set_title("Incoming electric field $E_i(x,y,t)$")

        # Draw PEC circle for reference
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(R*np.cos(theta), R*np.sin(theta), "k")

        def update(frame):
            t = frame * self.dt
            Ei = E_i(rho, t).reshape(X.shape)
            im.set_array(Ei)
            ax.set_title(f"$E_i(x,y)$ at t = {t:.2e} s")
            return im,

        ani = FuncAnimation(
            fig,
            update,
            frames=self.N_T,
            interval=interval,
            blit=False
        )

        plt.show()
        return ani


# ==============
# 6.1 VALIDATION
# ==============

def Q_6_1_validation(R, t_0, T, dt, t_end, N_S, N_G):

    phi = np.linspace(0, 2*np.pi, N_S + 1)
    curve = R * np.array([np.cos(phi), np.sin(phi)])

    t = np.arange(0, t_end, step=dt)
    N_T = len(t)

    mot = MOT(curve, dt, N_T, N_G)
    mot.plot_geometry()

    # incident pulse

    def E_i(rho, t):
        gamma = 4 / T * (mot.c * (t - t_0) - rho[mot._x])
        return 4 / T / np.sqrt(np.pi) * np.exp(-gamma**2)
    
    # V : (N_T, N_S)
    V = E_i(mot.rho, t.reshape((-1,1)))

    plt.plot(t, V[:,0],      label="shadow")
    plt.plot(t, V[:,N_S//2], label="sun")
    plt.xlabel("t (s)")
    plt.show(),

    mot.solve(V)
    omega, j = mot.positivespectrum()

    mot.plot_current_on_circle(time_index=np.argmin(np.abs(t - t_0)))
    mot.animate_current_on_circle1(R)
    mot.animate_current_1d()

    plt.figure()
    plt.plot(t, mot.U[0,:],      label="shadow")
    plt.plot(t, mot.U[N_S//2,:], label="sun")
    plt.xlabel("t (s)")
    plt.title("$dj/dt$")
    plt.legend()

    # ----------
    # analytical
    # ----------

    def j_z(phi,omega):
        k = omega.reshape(-1, 1, 1)/mot.c # rad/m 
        a = R                              # m
        n = np.arange(np.ceil(np.max(k)*a) + 2).reshape(1, -1, 1)
        return 1/1j/omega.reshape((-1, 1))/mot.mu * 2 * np.sum(1j**(n+1) * k * np.exp(1j*n*phi) / np.pi / k / a / fns.hankel2(n, k*a), axis=1)

    phi = np.arctan2(mot.rho[mot._y,:], mot.rho[mot._x,:])
    jz = j_z(phi, omega)

    # -------------
    # normalization
    # -------------

    # spectrum excitation
    A = np.exp(
        -1j * omega * t_0
        - (T * omega / (8 * mot.c))**2
    ) / mot.c

    plt.figure()
    plt.plot(omega/mot.c, np.abs(A))
    plt.title("spectrum excitation")
    plt.xlabel("$\\omega$/c (m$^{-1}$)")
    plt.xlim(0, 1)
    plt.ylabel("A (s/m)")

    # prevent j_0 from exploding for high omega
    A_abs = np.abs(A)
    A_max = np.max(A_abs)

    # Threshold parameter
    eps = 1e-3  # you can justify this in the report
    valid = A_abs >= eps * A_max
    omega_bad = omega[~valid]
    if omega_bad.size > 0:
        print("WARNING: excitation spectrum too small outside:")
        print(f"  omega > {omega_bad[0]:.3e} rad/s")

    j_0 = np.full_like(j, np.nan, dtype=float)
    # normalized current
    j_0[:, valid] = np.abs(
        j[:, valid] / A[valid].reshape(1, -1)
    )

    # -------------
    # visualization
    # -------------

    plt.figure()
    plt.plot(omega/mot.c, j_0[0,:],      '.', color='tab:blue',   label=f"numerical shadow")
    plt.plot(omega/mot.c, j_0[N_S//2,:], '.', color='tab:orange', label=f"numerical sun")

    plt.plot(omega/mot.c, jz[:,0],       '-', color='tab:blue',   label=f"analytical shadow")
    plt.plot(omega/mot.c, jz[:,N_S//2],  '-', color='tab:orange', label=f"analytical sun")

    plt.title("normalized current vs analytical")
    plt.xlabel("$\\omega/c$ (rad/m)")
    plt.xlim(0, 1)
    plt.ylabel("j$_0$")
    plt.legend()

    plt.figure()
    plt.plot(phi, j_0[:,1], '.', color='red',   label=f"$\\omega/c$={omega[1]/mot.c} rad/m")
    plt.plot(phi, j_0[:,2], '.', color='green', label=f"$\\omega/c$={omega[2]/mot.c} rad/m")
    plt.plot(phi, j_0[:,3], '.', color='blue',  label=f"$\\omega/c$={omega[3]/mot.c} rad/m")

    plt.plot(phi, jz[1], '-',    color='red',   label=f"$\\omega/c$={omega[1]/mot.c} rad/m")
    plt.plot(phi, jz[2], '-',    color='green', label=f"$\\omega/c$={omega[2]/mot.c} rad/m")
    plt.plot(phi, jz[3], '-',    color='blue',  label=f"$\\omega/c$={omega[3]/mot.c} rad/m")

    plt.title("normalized current vs analytical")
    plt.xlabel("$\\phi$ (rad)")
    plt.ylabel("j$_0$")
    plt.legend()

    plt.show()

Q_6_1_validation(
    R     = 10,     # [m] radius of PEC
    t_0   = 1e-7,   # [s] center of incident pulse
    T     = 20,     # [m] width of incident pulse
    dt    = 1e-9,   # [s] timestep
    t_end = 64e-8,  # [s] end of simulation
    N_S   = 32,     # [1] number of segments for PEC
    N_G   = 8       # [1] order of Gaussian quadrature
)


# ======================
# 6.2 Cylindrical Cavity
# ======================

def Q_6_2_cylindrical_cavity(R, dt, t_end, N_S, N_G):

    phi = np.linspace(0, 2*np.pi, N_S + 1)
    curve = R * np.array([np.cos(phi), np.sin(phi)])

    t = np.arange(0, t_end, step=dt)
    N_T = len(t)

    mot = MOT(curve, dt, N_T, N_G)
    mot.create_spacetimemesh_6_2(R, N_S, N_T)

    Ei2=mot.E_i2(mot.MeshR, mot.Mesht)

    radius= mot.radius
    mot.animate_E_i(radius,Ei2)
    U = mot.solve(V=Ei2.T)
    omega,j = mot.positivespectrum()
    #print(omega)
    # mot.analytical6_1_()
    # mot.plot61()
    #print(np.shape(mot.U))
    #print(np.shape(mot.jzanalyticaltime))


    mot.plot_current_on_circle(time_index=200, mode="vector")

    mot.animate_current_on_circle1(R, scale=2.0)
    ani1d = mot.animate_current_1d(interval=20)
    ani = mot.animate_Ei1(R, Ei2)
    ani2d = mot.animate_Ei1_2D(R, Ei2, interval=40)

    mot.analyticalzeros(10,20)

    print(mot.zeros)

Q_6_2_cylindrical_cavity(
    R     = 10,     # [m] radius of PEC
    dt    = 1e-9,   # [s] timestep
    t_end = 40e-8,  # [s] end of simulation
    N_S   = 80,     # [1] number of segments for PEC
    N_G   = 8       # [1] order of Gaussian quadrature
)


# ============
# 6.3 Creative
# ============
