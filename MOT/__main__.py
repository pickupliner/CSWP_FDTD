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
                 R=1.0, #Radius Pec
                 N_S=32,# spatial nodes
                 N_T=512, #temporal nodes
                 N_G=8, #amount of weights for quadrature
                 c=3e8,  #light constant
                 mu=np.pi * 4e-7 #permeability
                 ):

        # Physical constants
        self.c = c
        self.mu = mu
        self.R = R

        # Discretization
        self.N_S = N_S
        self.N_T = N_T
        self.N_G = N_G
        self.dt = np.pi * R / c  #timestep

        # Indices
        self._x, self._y = 0, 1  #whenever this is used first index is x-coordinates and second is y coordinates

        # Geometry
        self._build_geometry() # creates the geometry of the boundary which is  a (2,N_S) matrix with 0 x and 1 y

        # Time pulse parameters
        self._init_incident_pulse() #creates the values for the pulses

        # Quadrature
        self._init_quadrature() #creates  quadrature


        # Solution container
        self.U = np.zeros((self.N_S, self.N_T))  # solution is stored here

    # =====================
    # Geometry
    # =====================
    def curve(self, s):#m
        """
        curve creates the positional arguments for the circle
        
        :param self: class objext
        :param s: array
        """
        arg = 2 * np.pi * s #this is the circomference parametere more nodes means more discretiazation
        
        return self.R * np.array([np.cos(arg), np.sin(arg)]) #returns x and y for circle

    def _build_geometry(self):#m
        """
        _build_geometry creates the tangent lines and the coordinates of the circle 
        this is all on the boundary
        
        :param self: class objext
        """
        s = np.linspace(0, 1, self.N_S + 1)
        self.curve_points = self.curve(s)
        
        #these are all the tangent vectirs
        self.tangents = self.curve_points[:, 1:] - self.curve_points[:, :-1]
        #this gets all the norms of the tangents
        self.l = np.linalg.norm(self.tangents, axis=0)
        
        self.L = np.minimum(self.l, 2 * self.c * self.dt)
        # we check because under the sqrt it shouldnt be 0

    def rho_n(self, quadraturepoints): #m
        """
        these are the nodal points evaluated on the circumferance of the circle 
        the dimension is (2,Ns,Ng) when returned so on each segment there are 8 points that are integrated from
        on the nth segment (test function)?
        
        :param self: class object
        :param quadraturepoints: quadrature points 
        """
        shape = (2, self.N_S) + (1,) * len(quadraturepoints.shape)
        segmentnint =(
                    self.curve_points[:, :-1].reshape(shape)
                    + quadraturepoints * self.tangents.reshape(shape)
                     )
        
        return segmentnint

    def create_spacetimemesh(self):
        """
        creates mesh in spacetime You can return with recast such that 
        (2,N_s,m,n,N_rho,N_T,N_G)
        :param self: Description
        """
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
        """
        
        
        :param self: object model
        :param rho: where you evaluate the wave ()
        :param t: timesteps to evaluate (amount of t)
        """
        gamma = 4 / self.T * (self.c * (t - self.t_0) - rho[self._x])

        E_i = 4 / self.T / np.sqrt(np.pi) * np.exp(-gamma**2)
        
        return E_i

    # =====================
    # Quadrature
    # =====================
    def _init_quadrature(self):
        coordoriginal, woriginal = np.polynomial.legendre.leggauss(self.N_G)
        self.coordquad = (coordoriginal + 1) / 2
        self.weights = woriginal / 2

        self.coordquad = self.coordquad
        self.weights = self.weights

    def F(self, k, rho_m, rho_p):
        """
        Calculates integrand
        
        :param self: Description
        :param k: Wave number
        :param rho_m: Discretization nodes where we evaluate the E-field (2,N_S,1)
        :param rho_p: discretization source together with quadrature  (2,1,N_S,N_G)

        returns (N_s,N_s,N_G) shape
        """
        dist = np.linalg.norm(
            rho_m.reshape(rho_m.shape + (1,)) - rho_p,
            axis=0
        ) 
        distc= dist/self.c

        a = np.max([np.broadcast_to([k*self.dt], distc.shape), distc], axis=0)
        b = np.max([np.broadcast_to([(k+1)*self.dt], distc.shape), distc], axis=0)
        

        return np.log((b + np.sqrt(b**2 - distc**2)) /
                      (a + np.sqrt(a**2 - distc**2)))

    # =====================
    # Operators
    # =====================
    @functools.cache
    def Z(self, k):

        tmp = np.sqrt((2 * self.c * self.dt)**2 - self.L**2)
        Z0 = np.diag(
            -self.L / (2 * np.pi) * np.log((2 * self.c * self.dt + tmp) / self.L)
            - self.c * self.dt / np.pi * np.arctan(self.L / tmp)
        )

        m = np.arange(self.N_S).reshape(self.N_S, 1)
        n = np.arange(self.N_S).reshape(1, self.N_S)
        # (N_S,N_S)
        quad = np.sum(
            self.weights * self.F(k,
                            self.curve_points[:, m],
                            self.rho_n(self.coordquad)[:, n]),
            axis=-1 # over N_G
        )

        Z = -self.l[n] / (2 * np.pi) * quad
        Z = np.where((k == 0) & (m == n), Z0, Z)
        return Z

    def V1(self, j,E_i):
        return E_i(self.curve_points[:, :-1], j * self.dt)

    def V(self, j):
        
        return self.E_i1(self.curve_points[:, :-1], j * self.dt) #(N_s,)

    # =====================
    # Time-marching solver
    # =====================
    def solve(self,E_i=None):
        A = self.Z(0)
        b = -self.V(0) 
        self.U[:, 0], info = spla.gmres(A, b)
        assert info == 0 # error if equation could not be solved

        for j in range(1, self.N_T):
            # k = np.arange(0, j-1 + 1) # k = 0, .., j-1
            rhs = -self.V(j)
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
        u = self.dt * np.fft.rfft(self.U, axis=1)
        self.omega = 2 * np.pi * np.fft.rfftfreq(self.U.shape[1], self.dt)

        self.j = np.zeros_like(u, dtype=complex)
        self.j[:, 1:] = u[:, 1:] / (1j * self.omega[1:].reshape((1,-1))) / self.mu

        return self.omega, self.j


    # =====================
    # analytical
    # =====================
    def j_z(self,phi,omega):
        # omega = 1                   # rad TODO
        k = omega.reshape(-1, 1, 1)/self.c # rad/m TODO
        
        a = self.R                         # m

        n = np.arange(np.ceil(np.max(k)*a) + 2).reshape(1, -1, 1)
        return 1/1j/omega.reshape((-1, 1))/self.mu * 2 * np.sum(1j**(n+1) * k * np.exp(1j*n*phi) / np.pi / k / a / fns.hankel2(n, k*a), axis=1)
    

    def analytical6_1_(self):

        phi = np.linspace(-np.pi, np.pi, 128)
        self.jzanalyticalfrequency = self.j_z(phi,self.omega)
        Jz = np.abs(self.j_z(phi,self.omega))
        self.jzanalyticaltime = np.fft.irfft(self.jzanalyticalfrequency,axis=1)

        return phi,Jz
        
    # =====================
    # Visualization
    # =====================
    def plot_geometry(self):
        plt.plot(*self.curve_points)
        plt.axis("equal")
        plt.title("Geometry")
        plt.show()

    def plot61(self):
        plt.figure()
        plt.plot(np.arange(0, self.N_T*self.dt, self.dt), self.U[0,:])
        plt.plot(np.arange(0, self.N_T*self.dt, self.dt), self.U[self.N_S//2,:])
        plt.xlabel("t (s)")
        plt.title("j at phi=0")

        # --- Compute excitation spectrum ---
        A = np.exp(
            -1j * self.omega * self.t_0
            - (self.T * self.omega / (8 * self.c))**2
        ) / self.c
        
        A_abs = np.abs(A)
        A_max = np.max(A_abs)

        # Threshold parameter
        eps = 1e-3  # you can justify this in the report

        valid = A_abs >= eps * A_max
        omega_valid = self.omega[valid]

        print("Reliable frequency range:")
        print(f"  omega_min = {omega_valid[0]:.3e} rad/s")
        print(f"  omega_max = {omega_valid[-1]:.3e} rad/s")
        print(f"  (omega/c range = [{omega_valid[0]/self.c:.3e}, {omega_valid[-1]/self.c:.3e}] 1/m)")
        
        omega_bad = self.omega[~valid]
        if omega_bad.size > 0:
            print("WARNING: excitation spectrum too small outside:")
            print(f"  omega > {omega_bad[0]:.3e} rad/s")
        
        plt.figure()
        plt.plot(self.omega/self.c, np.abs(A))
        plt.title("spectrum excitation")
        plt.xlabel("$\\omega$/c (m$^{-1}$)")
        plt.ylabel("A (s/m)")
        j_0 = np.full_like(self.j, np.nan, dtype=float)
        print(np.shape(j_0))
        j_0[:, valid] = np.abs(
            self.j[:, valid] / A[valid].reshape(1, -1)
        )
        print(j_0)
        fig, axes = plt.subplots(2, 2, sharex='col')
        # FROM OMEGA[10] INSTABILITY STARTS TO DEVELOP AND ONLY BECOMES WORSE: TODO
        # instability due to divide by A: becomes nearly zero
        axes[0,0].plot(self.omega[:], j_0[0,:], label=f"{self.curve_points[[self._x, self._y],0]} m")
        axes[0,0].plot(self.omega[:], j_0[self.N_S//2,:], label=f"{self.curve_points[[self._x, self._y],self.N_S//2]} m")
        axes[0,0].set_title("normalized current")
        # axes[0,0].set_xlabel("$\\omega$ (rad/s)")
        axes[0,0].set_ylabel("j$_0$")
        axes[0,0].set_ylim(0, .03)
        axes[0,0].legend()

        axes[0,1].plot(np.arctan2(self.curve_points[self._y,:-1], self.curve_points[self._x,:-1]), j_0[:,1], label=f"$\\omega$={self.omega[1]} rad/s")
        axes[0,1].plot(np.arctan2(self.curve_points[self._y,:-1], self.curve_points[self._x,:-1]), j_0[:,2], label=f"$\\omega$={self.omega[2]} rad/s")
        axes[0,1].plot(np.arctan2(self.curve_points[self._y,:-1], self.curve_points[self._x,:-1]), j_0[:,3], label=f"$\\omega$={self.omega[3]} rad/s")
        axes[0,1].set_title("normalized current")
        # axes[0,1].set_xlabel("$\\phi$ (rad)")
        axes[0,1].set_ylabel("j$_0$")
        axes[0,1].legend()
        
        phi,jz = self.analytical6_1_()

        axes[1,1].plot(phi, jz[1], label=f"{self.omega[1]} rad/s")
        axes[1,1].plot(phi, jz[2], label=f"{self.omega[2]} rad/s")
        axes[1,1].plot(phi, jz[3], label=f"{self.omega[3]} rad/s")
        axes[1,1].set_title("analytical current")
        axes[1,1].set_xlabel("$\\phi$ (rad)")
        axes[1,1].set_ylabel("j$_z$")
        axes[1,1].legend()

        axes[1,0].plot(self.omega, jz[:,0], label=f"{self.curve_points[[self._x,self._y],0]} m")
        axes[1,0].plot(self.omega, jz[:,self.N_S//2], label=f"{self.curve_points[[self._x, self._y],self.N_S//2]} m")
        axes[1,0].set_title("analytical current")
        axes[1,0].set_xlabel("$\\omega$ (rad/s)")
        axes[1,0].set_ylabel("j$_z$")
        axes[1,0].legend()

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
        rho_mid = 0.5 * (self.curve_points[:, :-1] + self.curve_points[:, 1:])

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
    def animate_current_on_circle1(self, scale=1.0, interval=40):
        """
        Animate the surface current on the circular boundary.
        """

        # Element midpoints
        rho_mid = 0.5 * (self.curve_points[:, :-1] + self.curve_points[:, 1:])
        phi = np.arctan2(rho_mid[self._y], rho_mid[self._x])

        # Sort by angle
        idx = np.argsort(phi)
        phi = phi[idx]

        R = self.R

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
        # =====================
    # Animate 1D surface current vs angle
    # =====================
    def animate_current_1d(self, interval=30):
        """
        Animate the surface current along the circle in 1D
        (current magnitude vs angular position)
        """
        # Element midpoints
        rho_mid = 0.5 * (self.curve_points[:, :-1] + self.curve_points[:, 1:])
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
    def animate_Ei1(self, xlim=(-2, 2), interval=30):
        x = np.linspace(xlim[0]*self.R, xlim[1]*self.R, 400)
        rho = np.zeros((2, x.size))
        rho[0, :] = x

        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0, 1.2 * 4 / self.T / np.sqrt(np.pi))
        ax.set_xlabel("x (m)")
        ax.set_ylabel(r"$E_i$")
        ax.set_title("Incoming electric pulse")

        def update(frame):
            t = frame * self.dt
            Ei = self.E_i1(rho, t)
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
    def animate_Ei1_2D(self, interval=30):
        # Spatial grid
        x = np.linspace(-2*self.R, 2*self.R, 300)
        y = np.linspace(-2*self.R, 2*self.R, 300)
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
        ax.plot(self.R*np.cos(theta), self.R*np.sin(theta), "k")

        def update(frame):
            t = frame * self.dt
            Ei = self.E_i1(rho, t).reshape(X.shape)
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




mot = MOT(N_S=80, N_T=400)
Ei1= mot.E_i1(mot.curve_points[:, :-1],mot.dt)
mot.create_spacetimemesh()

Ei2=mot.E_i2(mot.MeshR,mot.Mesht)

#print(np.shape(Ei1))
#print(np.shape(Ei2))

radius= mot.radius
#mot.animate_E_i(radius,Ei1)
U = mot.solve()
omega,j = mot.positivespectrum()
#print(omega)
mot.analytical6_1_()
mot.plot61()
#print(np.shape(mot.U))
#print(np.shape(mot.jzanalyticaltime))


#mot.plot_current_on_circle(time_index=200, mode="vector")

#mot.animate_current_on_circle1(scale=2.0)
#ani1d = mot.animate_current_1d(interval=20)
##ani = mot.animate_Ei1()
#ani2d = mot.animate_Ei1_2D(interval=40)

mot.analyticalzeros(10,20)

print(mot.zeros)
