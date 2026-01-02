import numpy as np
import scipy
import scipy.special as fns
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import functools
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks


c=3e8
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
        assert np.all(np.abs(curve[:,0] - curve[:,-1]) < 1e-10) 
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
        dist_c = (np.linalg.norm(rho_resh - rhop_resh, axis=0)/self.c)
        #                                 k  m  n  g
        k = np.arange(self.N_T).reshape((-1, 1, 1, 1))
        # a,b : (k,m,n,g)
        a = np.maximum(k*self.dt, dist_c)
        b = np.maximum((k+1)*self.dt, dist_c)
        # F : (k,m,n,g)
        self.F = np.log((b + np.sqrt(b**2 - dist_c**2))/(a + np.sqrt(a**2 - dist_c**2)))

    def _init_Z(self):
        n = np.arange(self.N_S).reshape((1,1,-1))
        quad = np.einsum('g,kmng->kmn', self.weights, self.F)
        self.Z = -self.l[n]/2/np.pi * quad

        eps = 1e-20  # avoid divide by zero
        for m in range(self.N_S):
            root = np.sqrt((2*self.c*self.dt)**2 - self.L[m]**2 + eps)
            self.Z[0,m,m] = - self.L[m]/2/np.pi * np.log((2*self.c*self.dt + root)/self.L[m]) \
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
        R = int(input("what is radius"))
        zeros = []
        for n in range(totalnorder+1):
            zeros.append((self.c/R)*fns.jn_zeros(n,amountofzeros))

        self.zeros = zeros
        return zeros


    # =====================
    # Post-processing
    # =====================
    def positivespectrum(self):
        u = self.dt / np.sqrt(2*np.pi) * np.fft.rfft(self.U, axis=1)
        self.omega = 2 * np.pi * np.fft.rfftfreq(self.N_T, self.dt)

        self.j = np.zeros_like(u, dtype=complex)

        eps = 1e-12 * np.max(self.omega)
        valid = self.omega > eps

        self.j[:, valid] = u[:, valid] / (1j * self.omega[valid]) / self.mu
        self.Jt = np.fft.irfft(self.j, n=self.N_T, axis=1)

        return self.omega, self.j, self.Jt
    

    
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
    def plot_current_on_circle(self,curve, time_index, mode="polar"):
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
        rho_mid = 0.5 * (curve[:, :-1] + curve[:, 1:])

        # Angle of each element
        phi = np.arctan2(rho_mid[self._y], rho_mid[self._x])

        # Sort by angle for clean plotting
        idx = np.argsort(phi)
        phi = phi[idx]

        # Current at this time step
        j = self.Jt[:, time_index][idx]

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
        max_j = np.max(np.abs(self.Jt))
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
            j = self.Jt[:, frame][idx]
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
    def animate_current_on_curve(self,curve, scale=1.0, interval=40):
        """
        Animate surface current on an arbitrary closed curve
        by displacing points along the outward normal direction.
        """

        # Element midpoints
        rho_mid = 0.5 * (curve[:, :-1] + curve[:, 1:])  # (2, N_S)

        # Tangent vectors
        tangents = curve[:, 1:] - curve[:, :-1]
        tangents /= np.linalg.norm(tangents, axis=0, keepdims=True)

        # Normal vectors (rotate tangent by +90°)
        normals = np.vstack([-tangents[self._y], tangents[self._x]])

        # Normalize current
        max_j = np.max(np.abs(self.Jt))
        if max_j == 0:
            max_j = 1.0

        # Plot setup
        fig, ax = plt.subplots()
        ax.set_aspect("equal")

        pad = 0.3 * np.max(np.linalg.norm(rho_mid, axis=0))
        ax.set_xlim(rho_mid[0].min() - pad, rho_mid[0].max() + pad)
        ax.set_ylim(rho_mid[1].min() - pad, rho_mid[1].max() + pad)

        line, = ax.plot([], [], lw=2)

        # Keep animation reference
        self._ani = None

        def init():
            line.set_data(rho_mid[0], rho_mid[1])
            return line,

        def update(frame):
            j = self.Jt[:, frame]
            displacement = scale * j / max_j

            x = rho_mid[0] + displacement * normals[0]
            y = rho_mid[1] + displacement * normals[1]

            line.set_data(x, y)
            ax.set_title(f"Surface current, t = {frame*self.dt:.2e} s")
            return line,

        self._ani = FuncAnimation(
            fig,
            update,
            frames=self.N_T,
            init_func=init,
            interval=interval,
            blit=False
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
        max_j = np.max(np.abs(self.Jt))
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
            j = self.Jt[:, frame][idx]
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
    #animating the incoming E-field
    #============
    def animate_Ei1(self,curve, E_i=None, xlim=(-2, 2), interval=30):
        x = np.linspace(xlim[0]*np.max(curve), xlim[1]*np.max(curve), 400)
        rho = np.zeros((2, x.size))
        rho[0, :] = x

        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(-1, 1)
        ax.set_xlabel("x (m)")
        ax.set_ylabel(r"$E_i$")
        ax.set_title("Incoming electric pulse")

        # ---------- CASE 1: callable E_i ----------
        if callable(E_i):
            def update(frame):
                t = frame * self.dt
                Ei = E_i(rho, t)
                line.set_data(x, Ei)
                ax.set_title(f"Incoming electric pulse, t = {t:.2e} s")
                return line,

            frames = self.N_T

        # ---------- CASE 2: precomputed array ----------
        else:
            Ei_array = E_i
            frames = Ei_array.shape[1]

            def update(frame):
                line.set_data(x, Ei_array[:, frame])
                ax.set_title(f"Incoming electric pulse, t = {frame*self.dt:.2e} s")
                return line,

        ani = FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=interval,
            blit=False
        )

        plt.show()
        return ani

    def animate_Ei1_2D(self,curve, E_i=None, interval=30, grid_points=300):
        """
        Animate 2D incoming electric field in space (x,y) over time directly from r.
        """
        x_max = 2 * np.max(curve[0,:])
        y_max = 2 * np.max(curve[1,:])
        x = np.linspace(-x_max, x_max, grid_points)
        y = np.linspace(-y_max, y_max, grid_points)
        X, Y = np.meshgrid(x, y, indexing="ij")
        R_grid = np.sqrt(X**2 + Y**2)  # radial distance from center

        fig, ax = plt.subplots()
        im = ax.imshow(
            np.zeros_like(X),
            extent=[y.min(), y.max(), x.min(), x.max()],
            origin="lower",
            cmap="RdBu",
            vmin=-1, vmax=1,
            animated=True
        )
        ax.plot(self.curve[0,:], self.curve[1,:], "k")  # PEC boundary

        # ---------- CASE 1: E_i is callable ----------
        if callable(E_i):
            def update(frame):
                t = frame * self.dt
                Ei_r = E_i(np.stack([R_grid, np.zeros_like(R_grid)], axis=0), t)
                im.set_array(Ei_r)
                ax.set_title(f"$E_i(x,y)$ at t = {t:.2e} s")
                return im,
            frames = self.N_T

        # ---------- CASE 2: precomputed array ----------
        else:
            Ei_array = E_i
            assert Ei_array.shape[1] == self.N_T, "E_i array must have shape (N_rho, N_T)"
            # assume Ei_array already matches radial positions along self.radius
            def update(frame):
                # Evaluate at each R_grid point
                Ei_r = np.interp(R_grid.ravel(), self.radius, Ei_array[:, frame], left=0, right=0)
                Ei_r = Ei_r.reshape(R_grid.shape)
                im.set_array(Ei_r)
                ax.set_title(f"$E_i(x,y)$ at t = {frame*self.dt:.2e} s")
                return im,
            frames = self.N_T

        self._ani2D = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
        plt.show()
        return self._ani2D

def j_z(model,phi, omega,R):
        omega_safe = np.where(omega == 0, 1e-20, omega)  # avoid divide by zero
        k = omega_safe.reshape(-1, 1, 1) / c
        a = R
        # assuming k and a are defined
        N = int(np.ceil(np.max(k) * a) + 2)  # maximum |n| based on the decay rule

        # create n from -N to N
        n = np.arange(-N, N+1).reshape(1, -1, 1)  # shape (1, 2N+1, 1)
       
        print(n)
        return 1/1j/omega_safe.reshape((-1,1))/model.mu * 2 * np.sum(
            1j**(n+1) * k * np.exp(1j*n*phi) / np.pi / k / a / fns.hankel2(n, k*a), axis=1
        )
def analyticalvsnumerical(frequency,numerical,analytical,N_S,phi):
        plt.figure()
        plt.plot(frequency/c, numerical[0,:],      '.', color='tab:blue',   label=f"numerical shadow")
        plt.plot(frequency/c, numerical[N_S//2,:], '.', color='tab:orange', label=f"numerical sun")

        plt.plot(frequency/c, analytical[:,0],       '-', color='tab:blue',   label=f"analytical shadow")
        plt.plot(frequency/c, analytical[:,N_S//2],  '-', color='tab:orange', label=f"analytical sun")

        plt.title("normalized current vs analytical")
        plt.xlabel("$\\omega/c$ (rad/m)")
        plt.xlim(0, 1)
        plt.ylabel("j$_0$")
        plt.legend()

        plt.figure()
        plt.plot(phi, numerical[:,1], '.', color='red',   label=f"$\\omega/c$={frequency[1]/c} rad/m")
        plt.plot(phi, numerical[:,2], '.', color='green', label=f"$\\omega/c$={frequency[2]/c} rad/m")
        plt.plot(phi, numerical[:,3], '.', color='blue',  label=f"$\\omega/c$={frequency[3]/c} rad/m")

        plt.plot(phi, analytical[1], '-',    color='red',   label=f"$\\omega/c$={frequency[1]/c} rad/m")
        plt.plot(phi, analytical[2], '-',    color='green', label=f"$\\omega/c$={frequency[2]/c} rad/m")
        plt.plot(phi, analytical[3], '-',    color='blue',  label=f"$\\omega/c$={frequency[3]/c} rad/m")

        plt.title("normalized current vs analytical")
        plt.xlabel("$\\phi$ (rad)")
        plt.ylabel("j$_0$")
        plt.legend()

        plt.show()
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
    omega, j,jt = mot.positivespectrum()
    
    time_index = np.argmin(np.abs(mot.dt * np.arange(mot.N_T) - t_0))
    mot.plot_current_on_circle(curve, time_index, mode="angle")
    
    mot.animate_current_on_circle1(R)
    mot.animate_current_1d()
    mot.animate_Ei1(curve,E_i)
    mot.animate_Ei1_2D(curve,E_i)
    
    
    plt.figure()
    plt.plot(t, mot.U[0,:],      label="shadow")
    plt.plot(t, mot.U[N_S//2,:], label="sun")
    plt.xlabel("t (s)")
    plt.title("$dj/dt$")
    plt.legend()

    # ----------
    # analytical
    # ----------
    phi = np.arctan2(mot.rho[mot._y,:], mot.rho[mot._x,:])
    jz = j_z(mot,phi, omega,R)

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

    analyticalvsnumerical(omega,j_0,jz,N_S,phi)


def peakcomparison(model,omega,numerical):
    A = np.exp(
        -1j * omega * model.t_01
        - (model.T1 * omega / (8 * model.c))**2
    ) / model.c

    # --- normalize current to remove excitation
    A_abs = np.abs(A)
    A_max = np.max(A_abs)
    eps = 1e-3

    valid = A_abs > eps * A_max

    j_0 = np.full_like(numerical, np.nan, dtype=float)
    j_0[:, valid] = np.abs(numerical[:, valid] / A[valid])

    # --- angle-averaged spectrum
    spectrum = np.nanmean(j_0, axis=0)

    # --- find numerical peaks
    threshold = 0.25 * np.nanmax(spectrum)
    peaks, _ = find_peaks(spectrum, height=threshold)
    omega_numerical = omega[peaks]

    # --- analytical resonances (low order)
    zeros = model.analyticalzeros(5, 10)
    omega_analytical = np.concatenate(zeros)

    # ======================
    # Plot
    # ======================

    omega_max = np.max(omega)
    omega_analytical = omega_analytical[omega_analytical < omega_max]

    plt.figure(figsize=(8, 4))
    plt.plot(omega / model.c, spectrum, label="Numerical |j / A|")

    for w in omega_analytical:
        plt.axvline(
            w / model.c,
            color="k",
            alpha=0.5,
            lw=1.5
        )

    plt.scatter(
        omega_numerical / model.c,
        spectrum[peaks],
        color="red",
        zorder=3,
        label="Detected peaks"
    )

    plt.xlabel(r"$\omega / c$ (m$^{-1}$)")
    plt.ylabel("Normalized surface current")
    plt.title("Cylindrical cavity resonances")
    plt.legend()
    plt.xlim(0, omega_max / model.c)
    plt.tight_layout()
    plt.show()
    
# ======================
# 6.2 Cylindrical Cavity
# ======================
"""
Q_6_1_validation(
    R     = 10,     # [m] radius of PEC
    t_0   = 9e-8,   # [s] center of incident pulse
    T     = 20,     # [m] width of incident pulse
    dt    = 1e-9,   # [s] timestep
    t_end = 64e-8,  # [s] end of simulation
    N_S   = 32,     # [1] number of segments for PEC
    N_G   = 8       # [1] order of Gaussian quadrature
)
"""

# ======================
# 6.2 Cylindrical Cavity
# ======================
def J_surface(omega, phi, R, mu):
    """
    Compute analytical surface current on a PEC cylinder for a plane wave pulse.
    
    Parameters
    ----------
    omega : array, shape (N_ω,)
        Angular frequencies
    phi : array, shape (N_φ,)
        Angular positions along the cylinder
    R : float
        Cylinder radius
    mu : float
        Permeability of medium
    
    Returns
    -------
    J : array, shape (N_φ, N_ω)
        Surface current at each angle and frequency
    """
    import numpy as np
    import scipy.special as fns

    c = 3e8
    k = omega / c  # shape (N_ω,)

    # Determine N_max based on max(k*R)
    N_max = int(np.ceil(np.max(k)*R)) + 5
    n = np.arange(-N_max, N_max + 1)  # shape (2*N_max+1,)

    # Initialize J
    J = np.zeros((len(phi), len(omega)), dtype=complex)  # shape (N_φ, N_ω)

    # Sum over n
    for ni in n:
        H = fns.hankel2(ni, k*R)           # shape (N_ω,)
        exp_factor = np.exp(1j * ni * phi)  # shape (N_φ,)
        i_factor = 1j**(ni + 1)
        J += i_factor * exp_factor[:, None] / H[None, :]  # broadcast -> (N_φ, N_ω)

    # Apply prefactor
    J *= 2 / (1j * mu * omega[None, :]) / np.pi / R

    return J

def normalized_fft_and_plot(signal, incident, dt, title="Normalized MOT Spectrum"):
    """
    Compute and plot the normalized Fourier spectrum of a signal.

    Parameters
    ----------
    signal : array_like
        Numerical solution of the MOT system (e.g. x(t), v(t), fluorescence)
    incident : array_like
        Incident wave time series
    dt : float
        Time step
    title : str
        Plot title
    """

    N = len(signal)

    # Fourier transforms
    S = np.fft.fft(signal)
    I = np.fft.fft(incident)

    # Frequency axis
    freq = np.fft.fftfreq(N, dt)

    # Avoid division by zero
    eps = 1e-12
    normalized_spectrum = np.abs(S) / (np.abs(I) + eps)

    # Keep only positive frequencies
    mask = freq > 0
    freq = freq[mask]
    normalized_spectrum = normalized_spectrum[mask]

    # Plot
    plt.figure()
    plt.plot(freq, normalized_spectrum)
    plt.xlabel("Frequency")
    plt.ylabel("Normalized amplitude")
    plt.title(title)
    plt.grid(True)
    plt.show()

    return freq, normalized_spectrum
def Q_6_2_cylindrical_cavity(R, dt, t_end, N_S, N_G):

    phi = np.linspace(0, 2*np.pi, N_S + 1)
    curve = R * np.array([np.cos(phi), np.sin(phi)])
    
    # phi at segment midpoints (length N_S)
    phi_mid = 0.5 * (phi[:-1] + phi[1:])  # phi from curve nodes

    t = np.arange(0, t_end, step=dt)
    N_T = len(t)

    mot2 = MOT(curve, dt, N_T, N_G)
    mot2.create_spacetimemesh_6_2(R, N_S, N_T)
    # Evaluate Ei ONLY at boundary radius
    rho_boundary = R * np.ones((1, mot2.N_T))
    t_boundary   = mot2.t_01 + np.arange(mot2.N_T) * mot2.dt
    t_boundary   = t_boundary.reshape(1, -1)

    Ei_boundary = mot2.E_i2(rho_boundary, t_boundary)  # (1, N_T)

    # Replicate for each boundary segment
    V = np.tile(Ei_boundary.T, (1, mot2.N_S))  # (N_T, N_S)
    radius= mot2.radius
    #mot2.animate_E_i(radius,Ei_boundary)
    mot2.solve(V)
    omega,j,jt = mot2.positivespectrum()

    #peakcomparison(mot2,omega,j)
    jz= J_surface(omega,phi_mid,R,mot2.mu).T
    print(np.shape(jz),np.shape(j),np.shape(omega),np.shape(phi))
    analyticalvsnumerical(omega,j,jz,N_S,phi_mid)

    mot2.plot_current_on_circle(curve,time_index=200, mode="vector")

    mot2.animate_current_on_circle1(R, scale=2.0)
    mot2.animate_current_1d(interval=20)
    mot2.animate_Ei1(curve, Ei_boundary)
    #mot2.animate_Ei1_2D(curve, Ei_boundary, interval=40)

    mot2.analyticalzeros(10,20)
   

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
def square_curve(L=1.0, N_per_side=25):
    """
    Create a closed square curve centered at the origin.

    Parameters
    ----------
    L : float
        Side length of the square
    N_per_side : int
        Number of segments per side

    Returns
    -------
    curve : ndarray, shape (2, 4*N_per_side + 1)
        Closed square curve (first point == last point)
    """
    h = L / 2

    # Four sides (counter-clockwise)
    x1 = np.linspace(-h,  h, N_per_side, endpoint=False)
    y1 = -h * np.ones_like(x1)

    x2 =  h * np.ones(N_per_side)
    y2 = np.linspace(-h,  h, N_per_side, endpoint=False)

    x3 = np.linspace( h, -h, N_per_side, endpoint=False)
    y3 =  h * np.ones_like(x3)

    x4 = -h * np.ones(N_per_side)
    y4 = np.linspace( h, -h, N_per_side, endpoint=True)

    x = np.concatenate([x1, x2, x3, x4])
    y = np.concatenate([y1, y2, y3, y4])

    curve = np.vstack([x, y])

    # Explicitly close the curve
    curve = np.hstack([curve, curve[:, :1]])

    return curve

def Q_6_3_validation(L, t_0, T, dt, t_end,N_per_side,N_G):

    curve = square_curve(L, N_per_side)
    N_S= N_per_side*4
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
    peakcomparison(mot,omega,j)
    

    mot.animate_current_1d()
    mot.animate_Ei1(curve,E_i)
    mot.animate_Ei1_2D(curve,E_i)

    plt.figure()
    plt.plot(t, mot.U[0,:],      label="shadow")
    plt.plot(t, mot.U[N_S//2,:], label="sun")
    plt.xlabel("t (s)")
    plt.title("$dj/dt$")
    plt.legend()

    # ----------
    # analytical
    # ----------
    phi = np.arctan2(mot.rho[mot._y,:], mot.rho[mot._x,:])
    jz = j_z(mot,phi, omega,R)

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

    analyticalvsnumerical(omega,j_0,jz,N_S,phi)

"""Q_6_3_validation(
    L     = 10,     # [m] length of square PEC
    t_0   = 1e-7,   # [s] center of incident pulse
    T     = 20,     # [m] width of incident pulse
    dt    = 1e-9,   # [s] timestep
    t_end = 64e-8,  # [s] end of simulation
    N_per_side=20,   # [1]amount of segments per side of the PEC
    N_G   = 8       # [1] order of Gaussian quadrature
)
"""
def rounded_square_curve(
    L=10.0,          # side length
    r=1.0,           # corner radius
    N_side=20,       # points per straight side
    N_arc=20         # points per corner arc
):
    """
    Create a closed rounded-square curve centered at the origin.

    Returns
    -------
    curve : ndarray, shape (2, N+1)
        Closed curve suitable for MOT
    """
    assert r < L / 2, "Corner radius must be smaller than half the side length"

    h = L / 2 - r  # half straight length

    points = []

    # Helper to add arc
    def arc(cx, cy, theta0, theta1):
        theta = np.linspace(theta0, theta1, N_arc, endpoint=False)
        return np.vstack([
            cx + r * np.cos(theta),
            cy + r * np.sin(theta)
        ])

    # Bottom side
    x = np.linspace(-h, h, N_side, endpoint=False)
    y = -L/2 * np.ones_like(x)
    points.append(np.vstack([x, y]))

    # Bottom-right corner
    points.append(arc(h, -h, -np.pi/2, 0))

    # Right side
    y = np.linspace(-h, h, N_side, endpoint=False)
    x = L/2 * np.ones_like(y)
    points.append(np.vstack([x, y]))

    # Top-right corner
    points.append(arc(h, h, 0, np.pi/2))

    # Top side
    x = np.linspace(h, -h, N_side, endpoint=False)
    y = L/2 * np.ones_like(x)
    points.append(np.vstack([x, y]))

    # Top-left corner
    points.append(arc(-h, h, np.pi/2, np.pi))

    # Left side
    y = np.linspace(h, -h, N_side, endpoint=False)
    x = -L/2 * np.ones_like(y)
    points.append(np.vstack([x, y]))

    # Bottom-left corner
    points.append(arc(-h, -h, np.pi, 3*np.pi/2))

    curve = np.hstack(points)

    # Explicitly close curve
    curve = np.hstack([curve, curve[:, :1]])

    return curve

def Q_6_3rounded_validation(L, t_0, T, dt, t_end,N_per_side,N_G):

    curve = rounded_square_curve(L,0.5, N_per_side)
    N_S= N_per_side*4
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
    omega, j,jt= mot.positivespectrum()
    peakcomparison(mot,omega,j)
    
    mot.animate_current_on_curve(curve)
    mot.animate_current_1d()    
    mot.animate_Ei1(curve,E_i)
    mot.animate_Ei1_2D(curve,E_i)

    plt.figure()
    plt.plot(t, mot.U[0,:],      label="shadow")
    plt.plot(t, mot.U[N_S//2,:], label="sun")
    plt.xlabel("t (s)")
    plt.title("$dj/dt$")
    plt.legend()

    # ----------
    # analytical
    # ----------
    #phi = np.arctan2(mot.rho[mot._y,:], mot.rho[mot._x,:])
    #jz = j_z(mot,phi, omega,R)

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

    analyticalvsnumerical(omega,j_0,jz,N_S,phi)

"""
Q_6_3rounded_validation(
    L     = 10,     # [m] length of square PEC
    t_0   = 1e-6,   # [s] center of incident pulse
    T     = 20,     # [m] width of incident pulse
    dt    = 1e-9,   # [s] timestep
    t_end = 64e-8,  # [s] end of simulation
    N_per_side=20,   # [1]amount of segments per side of the PEC
    N_G   = 8       # [1] order of Gaussian quadrature
)
"""