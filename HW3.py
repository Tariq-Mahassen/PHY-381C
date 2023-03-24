import numpy as np
from scipy.integrate import solve_ivp

class AllenCahn:
    """
    Simulate the two-dimensional Allen-Cahn model using spectral methods
    """
    def __init__(self, nx, ny, d=1.0, kappa=1.0, Lx=1.0, Ly=1.0):
        self.nx, self.ny = nx, ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.d = d
        self.kappa = kappa
        
        kx = (np.pi / Lx) * np.hstack([np.arange(nx / 2 + 1), np.arange(1 - nx / 2, 0)])
        ky = (np.pi / Ly) * np.hstack([np.arange(ny / 2 + 1), np.arange(1 - ny / 2, 0)])
        self.kx, self.ky = kx, ky

        kxx, kyy = np.meshgrid(kx, ky)
        ksq = kxx**2 + kyy**2
        self.ksq = ksq.flatten()

        
    def _reaction(self, y):
        """
        Bistable reaction term: cast into real space, perform reaction, and then cast
        back into Fourier space.
        """
        ################################################
        #
        # Your code here. I recommend performing the reaction in real space, and using
        # the appropriate transformations to cast back and forth within the function.
        #
        ################################################
        y = np.reshape(y, (self.ny, self.nx))
        yh = np.fft.ifft2(y)
        out = np.fft.fft2(yh * (1 - yh**2))
        return out.flatten()

    def rhs(self, t, y):
        """
        For technical reasons, this function needs to take a one-dimensional vector, 
        and so we have to reshape the vector back into the mesh
        """
        return self.kappa * self._reaction(y) - self.d * self.ksq  * y


    def solve(self, y0, t_min, t_max, nt, **kwargs):
        """
        Solve the heat equation using the odeint solver
        **kwargs are passed to scipy.integrate.solve_ivp
        """
        tpts = np.linspace(t_min, t_max, nt)
        y0_k = np.fft.fft2(y0)

        out = solve_ivp(self.rhs, (t_min, t_max), y0_k.flatten(), t_eval=tpts, **kwargs)
        sol = out.y.T
        tpts =  out.t

        ## Convert back to real space
        sol2 = list()
        for row in sol:
            sol2.append(np.fft.ifft2(row.reshape((self.nx, self.ny))))
        sol2 = np.array(sol2)
        print(f"Sanity check: imaginary residual is: {np.mean(np.abs(np.imag(np.array(sol2))))}")
        sol2 = np.real(sol2)
        sol = sol2

        return tpts, sol
      

def make_initial_conditions(n):
    """
    Make a bump-shaped initial conditions array
    """
    xx, yy = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    rr =  (xx**2 + yy**2)**0.5
    u_ic = 1 - 0.5 * np.copy(np.exp(-rr**2 - xx**2))
    v_ic = 0.25 * np.copy(np.exp(-rr**2))
    return u_ic, v_ic

class GrayScott:
    """
    Simulate the two-dimensional Gray-Scott model
    Parameters
        nx (int): number of grid points in the x direction
        ny (int): number of grid points in the y direction
        Lx (float): length of the domain in the x direction
        Ly (float): length of the domain in the y direction
        du (float): diffusion coefficient for u
        dv (float): diffusion coefficient for v
        kappa (float): degradation rate of v
        b (float): growth rate of u
    """

    def __init__(self, nx, ny, du=0.1, dv=0.05, b=0.0545, kappa=0.1165, Lx=1.0, Ly=1.0):
        self.nx, self.ny = nx, ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.du, self.dv = du, dv
        self.kappa = kappa
        self.b = b
        
        ## We need to define a mesh for the frequency domain
        kx = (2 * np.pi / Lx) * np.hstack([np.arange(nx / 2 + 1), np.arange(1 - nx / 2, 0)]) / nx
        ky = (2 * np.pi / Ly) * np.hstack([np.arange(ny / 2 + 1), np.arange(1 - ny / 2, 0)]) / ny
        self.kx, self.ky = kx, ky
        kxx, kyy = np.meshgrid(kx, ky)

        ksq = kxx**2 + kyy**2
        self.ksq = ksq

        
    def _reaction(self, y):
        """
        Compute the reaction term in real space
        Args:
            y (np.ndarray): array of shape (2 * nx * ny, ) containing the two fields
                u and v, stacked together
        """
        u, v = y[:(self.ny * self.nx)], y[-(self.ny * self.nx):]
        uv2 = u * (v**2)
        rxn_u = -uv2 + self.b * (1 - u)
        rxn_v = uv2 - self.kappa * v
        y_out = np.hstack([rxn_u, rxn_v])
        return y_out

    # ## Finite difference approach
    # def _laplace(self, y):
    #     """
    #     Calculate the Laplacian in real space of a flat array

    #     Args:
    #         y (np.ndarray): array of shape (nx * ny, ) containing the a single field
    #     """
    #     y = np.reshape(y, (self.nx, self.ny))
    #     lap = np.zeros((self.ny, self.nx))
    #     # enforce reflection boundary conditions by padding rows and columns
    #     y = np.vstack([y[-1, :][None, :], y, y[0, :][None, :]])
    #     y = np.hstack([y[:, -1][:, None], y, y[:, 0][:, None]])

    #     # calculate vectorized laplace operator
    #     lap = y[:-2, 1:-1] + y[1:-1, :-2] + y[2:, 1:-1] + y[1:-1, 2:]
    #     lap  -= 4 * y[1:-1, 1:-1]
        
    #     #lap /= self.dx * self.dy
    #     return lap.flatten()


    def _laplace(self, y):
        """
        Calculate the Laplacian in Fourier space
        """
        yk = np.fft.fft2(y)
        lap = -self.ksq * yk
        lap = np.fft.ifft2(lap)
        return np.real(lap).flatten()


    def _diffusion(self, y):
        """
        Calculate the diffusion term in Fourier space
        Args:
            y (np.ndarray): array of shape (2 * nx * ny, ) containing the two fields
                u and v, stacked together
        """
        u, v = y[:(self.ny * self.nx)], y[-(self.ny * self.nx):]
        lap_u = self._laplace(u)
        lap_v = self._laplace(v)
        u_out = self.du * lap_u
        v_out = self.dv * lap_v
        y_out = np.hstack([u_out, v_out])
        return y_out



    def rhs(self, t, y):
        """
        For technical reasons, this function needs to take a one-dimensional vector, 
        and so we have to reshape the vector back into the mesh
        """
        out = self._reaction(y) + self._diffusion(y)
        return out


    def solve(self, y0, t_min, t_max, nt, **kwargs):
        """
        Solve the heat equation using the solve_ivp solver
        Args:
            y0 (np.ndarray): initial condition
            t_min (float): minimum time
            t_max (float): maximum time
            nt (int): number of time steps
            **kwargs: keyword arguments to pass to solve_ivp
        """
        u0, v0 = y0
        tpts = np.linspace(t_min, t_max, nt)
        y0 = np.hstack([u0.flatten(), v0.flatten()])
        out = solve_ivp(self.rhs, (t_min, t_max), y0, t_eval=tpts, **kwargs)
        sol = out.y.T
        
        u, v = sol[:, :self.nx * self.ny], sol[:, self.nx * self.ny:]
        u = np.reshape(u, (nt, self.ny, self.nx))
        v = np.reshape(v, (nt, self.ny, self.nx))
        return tpts, np.stack([u, v], axis=-1)
