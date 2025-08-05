import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
from scipy.integrate import solve_ivp

import gpytorch
from gpytorch.models import ExactGP
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.means import MultitaskMean, ConstantMean
from gpytorch.kernels import MultitaskKernel, RBFKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


class TorchBiosystemModelWithInputs(nn.Module):
    def __init__(self, K_m, s_e, mu_max, method="rk4"):
        super().__init__()
        self.register_buffer("K_m", torch.tensor(K_m))
        self.register_buffer("s_e", torch.tensor(s_e))
        self.register_buffer("mu_max", torch.tensor(mu_max))
        assert method in ("rk4", "euler")
        self.method = method

    def _rhs(self, y, D):
        # y: (...,2), D: (...,) broadcastable
        b, s = y[..., 0], y[..., 1]
        rho  = self.mu_max * s / (self.K_m + s)
        dbdt = -D * b      + rho * b
        dsdt =  D * (self.s_e - s) - rho * b
        return torch.stack([dbdt, dsdt], dim=-1)

    def step(self, y, dt, D):
        """
        Advance the system by one time step.
        
        y: (...,2) current state
        dt: scalar step size
        D: scalar or broadcastable with y batch dims

        returns: (...,2) state at t+dt
        """
        if self.method == "euler":
            k1 = self._rhs(y, D)
            y_next = y + dt * k1
        else:  # RK4
            k1 = self._rhs(y, D)
            k2 = self._rhs(y + dt/2 * k1, D)
            k3 = self._rhs(y + dt/2 * k2, D)
            k4 = self._rhs(y + dt   * k3, D)
            y_next = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return y_next



#===Neural Network Class
class Neuralnetwork (nn.Module):
    def __init__(self,dim):
        super(Neuralnetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dim+1, 128),
            nn.ReLU(),              
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dim)      
        )
    def forward(self, x):
        return self.model(x)


# === Batched Neural ODE function ===

"""Neural ODE that takes the state dimension(=2) and input dimension (=1), as well as the time from which the data stems.
The model is trained by approximating dx/dt=f(x,u), integrating it and then comparing it to the 'real' states. 
When using multiple training sets, training can get quite time intensive, so it is easier to create a 'batch' of all the data, where 
one can simply stack all the elements in a List containing [B,S,U]_1,....[B,S,U]_N, where N is the number of batches"""
import numbers
class ODEFunc(nn.Module):
    def __init__(self, dim_x, dim_u, t_grid, u_seq):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x + dim_u, 64),
            nn.Tanh(),
            nn.Linear(64, dim_x),
            
        )
        self.t_grid = t_grid        # [T]
        self.u_seq = u_seq          # [N, T, 1]

    def get_u_at_t(self, t):
        if isinstance(self.u_seq, numbers.Number):
            return torch.tensor([[self.u_seq]], dtype=torch.float32)  # shape [1, 1] e.g. for handing over scalars
        # If single trajectory, shape might be [T, 1]
        if isinstance(self.u_seq, torch.Tensor):
            if self.u_seq.dim() == 2:
                # Assume shape [T, 1]as e.g. a nparray
                idx = torch.argmin(torch.abs(self.t_grid - t))
                return self.u_seq[idx].unsqueeze(0)  # shape [1, 1]
            else:
                # Batch case: [N, T, 1]
                idx = torch.argmin(torch.abs(self.t_grid - t))
                return self.u_seq[:, idx, :]  # [N, 1]
        raise ValueError("u_seq must be a scalar or a torch.Tensor")

    def forward(self, t, x):
        """
        x: [N, dim_x]
        Returns dx: [N, dim_x]
        i added a singe_traj check, because later, always having to carry around the batchdim made it suuuper tedious to check 
        for correct dimensions in Matrix multiplications (e.g EKF)
        """
        single_traj = False
        # --- If x is 1D (single state vector), make it [1, dim_x]
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_traj = True

        # --- Get the input u(t)
        u_t = self.get_u_at_t(t)   # [N, 1]
        if not torch.is_tensor(u_t):
            u_t = torch.tensor(u_t, dtype=torch.float32)
        xu = torch.cat([x, u_t], dim=-1)  # [N, dim_x + dim_u]
        dx = self.net(xu)  # [N, dim_x]

        # --- If single trajectory, remove the fake batch dim
        if single_traj:
            dx = dx.squeeze(0)  # back to [dim_x]

        return dx

#===Multi-output-Gaussian Process
class MultitaskGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        # MultitaskMean wraps a base mean for each task
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=2)
        # MultitaskKernel wraps a base RBF for each task, with shared structure
        self.covar_module = MultitaskKernel(RBFKernel(), num_tasks=2, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)      # → [N, 2]
        covar_x = self.covar_module(x)    # LazyTensor representing interleaved [N*2, N*2]
        return MultitaskMultivariateNormal(mean_x, covar_x)