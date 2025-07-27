import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
from scipy.integrate import solve_ivp

class BiosystemModel:
    def __init__(self, K_m, s_e, mu_max, D_func=None):
        self.K_m = K_m
        self.s_e = s_e
        self.mu_max = mu_max
        self.D_func = D_func if D_func is not None else (lambda t: 0.0)  # default: D=0

    def rhs(self, t, y):
        D = self.D_func(t)  # always use the current D_func
        b, s = y
        rho = self.mu_max * s / (self.K_m + s)
        dbdt = -D * b + rho * b
        dsdt = D * (self.s_e - s) - rho * b
        return [dbdt, dsdt]

    def set_D_func(self, D_func):
        """Set a new input function D(t) for future simulations."""
        self.D_func = D_func

    def __call__(self, t_eval, y0=None, D_func=None):
        """
        Solve the system over time vector t_eval.
        - y0: initial conditions [b0, s0]. Defaults to [0.1, 5.0].
        - D_func: optional new D(t) to use *just for this solve*.
        """
        # Temporarily override D_func if given
        if D_func is not None:
            old_func = self.D_func
            self.D_func = D_func

        if y0 is None:
            y0 = [0.1, 5.0]

        t_span = (t_eval[0], t_eval[-1])
        sol = solve_ivp(self.rhs, t_span, y0, t_eval=t_eval)

        # Restore original D_func if we overrode it
        if D_func is not None:
            self.D_func = old_func

        return sol
    
# === Batched Neural ODE function ===

"""Neural ODE that takes the state dimension(=2) and input dimension (=1), as well as the time from which the data stems.
The model is trained by approximating dx/dt=f(x,u), integrating it and then comparing it to the 'real' states. 
When using multiple training sets, training can get quite time intensive, so it is easier to create a 'batch' of all the data, where 
one can simply stack all the elements in a List containing [B,S,U]_1,....[B,S,U]_N, where N is the number of batches"""
class ODEFunc_input(nn.Module):
    def __init__(self, dim_x, dim_u, t_grid, u_seq):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x + dim_u, 64),
            nn.Tanh(),
            nn.Linear(64, dim_x)
        )
        self.t_grid = t_grid        # [T]
        self.u_seq = u_seq          # [N, T, 1]

    def get_u_at_t(self, t):
        """
        Returns u(t) for ALL trajectories at once (zero-order hold).
        t: scalar
        """
        idx = torch.argmin(torch.abs(self.t_grid - t))
        return self.u_seq[:, idx, :]    # [N, 1]

    def forward(self, t, x):
        """
        x: [N, dim_x]
        Returns dx: [N, dim_x]
        """
        u_t = self.get_u_at_t(t)                # [N, 1]
        xu = torch.cat([x, u_t], dim=-1)        # [N, dim_x + dim_u]
        dx = self.net(xu)                       # [N, dim_x]
        return dx