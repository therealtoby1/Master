######################Neural Networks###########################################
import torch
import torch.nn as nn
import gpytorch
import math

"""simple neural network, nothing special"""
class Neuralnetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, 128),nn.ReLU(),
            nn.Linear(128, 128),nn.ReLU(),
            nn.Linear(128, 128),nn.ReLU(),
            nn.Linear(128, 128),nn.ReLU(),
            nn.Linear(128, dim)
        )
    def forward(self, x):   
        return self.model(x)
    

""""extra emphasis on the Input by creating another hidden layer for it first and then adding it to the other layer"""
class Input_Enhance_Net(nn.Module):
    def __init__(self, M, H=128,dim=101):
        super().__init__()
        self.M=M
        self.enc_n = nn.Sequential(nn.Linear(M, H), nn.ReLU())
        self.enc_s = nn.Sequential(nn.Linear(1, H), nn.ReLU())          # saturating path for S
        
        self.block  = nn.Sequential(
            nn.Linear(H,128), nn.ReLU(),
            nn.Linear(128, 128),nn.ReLU(),
            nn.Linear(128, 128),nn.ReLU(),
            nn.Linear(128, dim)
        )

    def forward(self, x):
        # x = [n(M), S]
        n, S = x[..., :self.M], x[..., self.M:self.M+1]
        h = self.enc_n(n) + self.enc_s(S)  # S and u injected at hidden width
        return self.block(h)
    
class FiLM_Net(nn.Module):
    def __init__(self, M, H=128,dim=101):
        super().__init__()
        self.M = M

        # Encoder for n
        self.enc_n = nn.Sequential(
            nn.Linear(M, H),
            nn.ReLU()
        )

        # FiLM generator for S
        self.film_gen = nn.Sequential(
            nn.Linear(1, H),
            nn.ReLU(),
            nn.Linear(H, 2*H)   # outputs [gamma, beta]
        )

        # Main block
        self.block = nn.Sequential(
            nn.Linear(H, 128), nn.ReLU(),
            nn.Linear(128, 128),nn.ReLU(),
            nn.Linear(128, 128),nn.ReLU(),
            nn.Linear(128, dim)
        )

    def forward(self, x):
        # Split input
        n, S = x[..., :self.M], x[..., self.M:self.M+1]

        # Encode n
        h = self.enc_n(n)   # (batch, H)

        # Generate FiLM params from S
        film_params = self.film_gen(S)         # (batch, 2H)
        gamma, beta = film_params.chunk(2, dim=-1)  # each (batch, H)

        # Apply FiLM modulation
        h = gamma * h + beta

        # Continue with main block
        return self.block(h)
    
#Resnet
"""Residual Neural Network to generate a mapping x_k+1=x+deltax"""
class ResidualBlock(nn.Module):         #building bricks for ResNet
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x):
        return x + self.block(x)   # residual connection
    
class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=4):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # stack residual blocks
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h = torch.relu(self.input_layer(x))
        h = self.blocks(h)
        return self.output_layer(h)
    
#model4 = ResNet(input_dim=M+1, hidden_dim=128, output_dim=M+1, num_blocks=4)
"""Fourier Neural Networks to learn the Mapping operator"""
# -----------------------------
# 1. Fourier Layer
# -----------------------------
class SpectralConv1d(nn.Module):
    """
    1D Fourier layer: FFT -> linear transform on low modes -> IFFT
    """
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # number of Fourier modes to keep

        # complex weights for low-frequency modes
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat) /
            (in_channels * out_channels)
        )

    def compl_mul1d(self, input, weights):
        # input: (batch, in_channel, modes)
        # weights: (in_channel, out_channel, modes)
        return torch.einsum("bim,iom->bom", input, weights)

    def forward(self, x):
        # x: (batch, in_channels, N)
        batchsize, channels, N = x.shape

        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)  # (batch, in_channels, N//2+1)

        # keep only first self.modes
        out_ft = torch.zeros(batchsize, self.out_channels, N//2 + 1,
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(
            x_ft[:, :, :self.modes], self.weights
        )

        # inverse FFT
        x = torch.fft.irfft(out_ft, n=N, dim=-1)  # (batch, out_channels, N)
        return x


# -----------------------------
# 2. Base FNO model
# -----------------------------
class FNO1d(nn.Module):
    def __init__(self, modes, width, in_channels=1, out_channels=1, depth=4):
        """
        modes: number of Fourier modes kept
        width: latent channel size
        depth: number of Fourier layers
        """
        super().__init__()
        self.modes = modes
        self.width = width
        self.depth = depth

        self.fc0 = nn.Linear(in_channels, width)

        self.fourier_layers = nn.ModuleList()
        self.ws = nn.ModuleList()
        for _ in range(depth):
            self.fourier_layers.append(SpectralConv1d(width, width, modes))
            self.ws.append(nn.Conv1d(width, width, 1))  # local linear

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x: (batch, N, in_channels)
        x = self.fc0(x)  # (batch, N, width)
        x = x.permute(0, 2, 1)  # (batch, width, N)

        for i in range(self.depth):
            x1 = self.fourier_layers[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            x = torch.relu(x)

        x = x.permute(0, 2, 1)  # (batch, N, width)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)  # (batch, N, out_channels)
        return x


# -----------------------------
# 3. Wrapper for your dataset format
# -----------------------------
class FNO1dWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        # x: (batch, M+1)   -> from your dataset
        x = x.unsqueeze(-1)            # -> (batch, M+1, 1)
        y = self.base(x)               # -> (batch, M+1, 1)
        return y.squeeze(-1)           # -> (batch, M+1)

# example instantiating
# base_fno = FNO1d(modes=16, width=64, in_channels=1, out_channels=1, depth=4)


#Variational Sparse LMC (Mulit-output) Gaussian Process

# --- Likelihood for multitask regression


class SparseLMCMultitaskGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_tasks: int, num_latents: int = 3):
        """
        inducing_points: (num_latents, M, D)
        """
        
        # Variational distribution & strategy over inducing points (batched by num_latents)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2),  # M
            batch_shape=torch.Size([inducing_points.size(0)])  # = num_latents
        )
        base_variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,                      # (num_latents, M, D)
            variational_distribution,
            learn_inducing_locations=True,
        )

        # Wrap with LMC to mix latents -> tasks
        lmc_strategy = gpytorch.variational.LMCVariationalStrategy(
            base_variational_strategy,
            num_tasks=num_tasks,
            num_latents=inducing_points.size(0),  # = num_latents
            latent_dim=-1,                        # latents live in the leading batch dim
        )
        super().__init__(lmc_strategy)

        # Mean & kernel must also be batched by num_latents
        batch_shape = torch.Size([inducing_points.size(0)])   # = [num_latents]
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(-1), batch_shape=batch_shape),
            batch_shape=batch_shape
        )

    def forward(self, x):
        # mean_x, covar_x each have batch_shape = [num_latents]
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

###########################PDE- EQUATION################################################################
def build_P_torch(m, q, Bqq, delta_m):
    """
    Build probability matrix P for size partitioning.
    m: (M,) torch tensor
    """
    M = m.shape[0]
    P = torch.zeros((M, M), dtype=torch.float32)

    for k in range(M):
        mk = m[k]
        if mk <= 0:
            continue

        x = m[:k] / mk  # daughters
        nonzero_vals = (1.0 / Bqq) * (1.0 / mk) * (x ** (q - 1)) * ((1 - x) ** (q - 1))
        P[k, :k] = nonzero_vals

        # Normalize
        Z = torch.sum(P[k, :k]) * delta_m
        if Z > 0:
            P[k, :k] /= Z

        # First moment condition
        mom = torch.sum(m[:k] * P[k, :k]) * delta_m
        if mom > 0:
            P[k, :k] *= (0.5 * mk / mom)

    return P



def r_g_phys(m, S_hat, S_max, ks_max=0.8, K_s=2.0):
    S = S_max * S_hat
    rho = (ks_max * S) / (K_s + S + 1e-12)
    return rho * m

def Gamma_phys(m, S_hat, m_max, m_min_div, S_max, ks_max=0.8, K_s=2.0):
    eps = 1e-12
    gamma_m = 1.0/(m_max - m + eps) - 1.0/(m_max - m_min_div + eps)
    # physical: no division below threshold -> clip to 0
    gamma_m = torch.clamp(gamma_m, min=0.0)
    return gamma_m * r_g_phys(m, S_hat, S_max, ks_max, K_s)

class PopulationSystemNormalizedSplit(nn.Module):
    """
    Normalized state x = [n_hat (M,), S_hat].
    Exact dynamics & boundaries; per micro-step we do:
      (A) transport (explicit upwind, CFL)
      (B) division+dilution (semi-implicit sink, explicit gain)  -> positivity-preserving
      (C) substrate explicit.
    """
    def __init__(self, m, P, delta_m,
                 se_phys, m_max, minimal_division_mass,
                 n_max, S_max,
                 ks_max=0.8, K_s=2.0,
                 delta_t=0.01,
                 cfl_limit=0.8, react_limit=0.5,
                 integrator="rk4",              # used only for transport part
                 dtype=torch.float32,
                 project_nonneg=True,           # clamp tiny negatives after substep
                 eps_clip=0.0):                 # set e.g. 0 or 1e-12
        super().__init__()
        self.register_buffer("m", m.to(dtype))
        self.register_buffer("P", P.to(dtype))
        self.delta_m = float(delta_m)

        self.se_phys = float(se_phys)
        self.m_max   = float(m_max)
        self.m_min   = float(minimal_division_mass)
        self.ks_max  = float(ks_max)
        self.K_s     = float(K_s)

        self.n_max   = float(n_max)
        self.S_max   = float(S_max)
        self.kappa   = self.n_max / self.S_max
        self.se_hat  = self.se_phys / self.S_max

        self.delta_t = float(delta_t)
        self.cfl_limit   = float(cfl_limit)
        self.react_limit = float(react_limit)
        self.integrator  = integrator.lower()
        self._dtype      = dtype
        self.project_nonneg = project_nonneg
        self.eps_clip       = float(eps_clip)

        # faces for transport fluxes
        m_faces = 0.5 * (self.m[1:] + self.m[:-1])
        self.register_buffer("m_faces", m_faces)

    # ---------- transport divergence ONLY: -∂_m( r_g * n_hat ) ----------
    def transport_rhs(self, n_hat, S_hat):
        # alpha(S) in physical units
        S_phys = self.S_max * S_hat
        alpha = self.ks_max * S_phys / (self.K_s + S_phys)

        # growth velocity * mass
        rg = alpha * self.m

        # rg * n_hat = growth flux
        rg_n = rg * n_hat
        flux_out = rg_n[-1]         # store last-bin flux
        rg_n = rg_n.clone()
        rg_n[-1] = 0.0              # enforce zero at the right boundary

        # derivative in m
        d_rg_n_dm = torch.zeros_like(n_hat)
        d_rg_n_dm[1:] = (rg_n[1:] - rg_n[:-1]) / self.delta_m
        d_rg_n_dm[0]  = rg_n[0] / self.delta_m   # left boundary

        # transport contribution
        T = -d_rg_n_dm
        T[-1] += flux_out / self.delta_m         # right boundary correction

        return T, rg

    # ---------- one micro-step (A,B,C) ----------
    def _micro_step(self, x, h, t=0.0, D_t=0.0):
        n_hat = x[:-1]
        S_hat = x[-1]

        # (A) transport explicit (with RK4/Euler on transport part only)
        def tr_rhs(nv):
            T, _ = self.transport_rhs(nv, S_hat)
            return T

        if self.integrator == "euler":
            n_tmp = n_hat + h * tr_rhs(n_hat)
        else:
            k1 = tr_rhs(n_hat)
            k2 = tr_rhs(n_hat + 0.5*h*k1)
            k3 = tr_rhs(n_hat + 0.5*h*k2)
            k4 = tr_rhs(n_hat + h*k3)
            n_tmp = n_hat + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # (B) division + dilution: semi-implicit sink, explicit gain
        G = Gamma_phys(self.m, S_hat, self.m_max, self.m_min, self.S_max, self.ks_max, self.K_s)
        gain = 2.0 * self.delta_m * (G * n_tmp) @ self.P
        denom = (1.0 + h * (G + D_t))                 # elementwise
        n_next = (n_tmp + h * gain) / denom

        # (C) substrate explicit using n_tmp (after transport)
        v = r_g_phys(self.m, S_hat, self.S_max, self.ks_max, self.K_s)
        uptake = torch.sum(v * n_tmp) * self.delta_m
        S_next = S_hat + h * (- self.kappa * uptake + D_t * (self.se_hat - S_hat))

        if self.project_nonneg:
            n_next = torch.clamp(n_next, min=self.eps_clip)
            S_next = torch.clamp(S_next, min=self.eps_clip)

        return torch.cat([n_next, S_next.view(1)])

    # ---------- number of micro-steps from transport & reaction scales ----------
    def _num_substeps(self, x, D_t):
        S_hat = float(x[-1])
        S = self.S_max * S_hat
        alpha = self.ks_max * S / (self.K_s + S + 1e-12)
        v_max = abs(alpha) * float(self.m.max())
        cfl_ratio = v_max * self.delta_t / (self.delta_m + 1e-12)
        N_cfl = 1 if (v_max == 0.0 or cfl_ratio <= self.cfl_limit) else int(math.ceil(cfl_ratio/self.cfl_limit))

        # reaction timescale: max(Gamma + D)
        rgv  = r_g_phys(self.m, S_hat, self.S_max, self.ks_max, self.K_s)
        G    = (1.0/(self.m_max - self.m + 1e-12) - 1.0/(self.m_max - self.m_min + 1e-12))
        G    = torch.clamp(G, min=0.0) * rgv
        rate = float(torch.max(G).item() + abs(D_t))
        react_ratio = rate * self.delta_t
        N_react = 1 if (rate == 0.0 or react_ratio <= self.react_limit) else int(math.ceil(react_ratio/self.react_limit))

        return max(1, N_cfl, N_react)

    # ---------- one UKF step with adaptive micro-steps & external D(t) ----------
    def step(self, x, t=0.0, i=0.0, D_func=None):
        D_t = float(D_func(t, i)) if D_func is not None else 0.0
        N   = self._num_substeps(x, D_t)
        h   = self.delta_t / N
        xk  = x
        for s in range(N):
            xk = self._micro_step(xk, h, t + s*h, D_t)
        return xk

    def forward(self, x, u=None):  # u = D_t if you pass via U_seq
        if u is None:
            return self.step(x)
        return self.step(x, D_func=(lambda t, i, D=u: D))


from torchdiffeq import odeint

class PopulationPDE(nn.Module):
    def __init__(self, m, P, delta_m, se_phys, m_max, m_min, 
                 n_max, S_max, ks_max=0.8, K_s=2.0):
        super().__init__()
        self.register_buffer("m", m)
        self.register_buffer("P", P)
        self.delta_m = float(delta_m)
        self.se_phys = float(se_phys)
        self.m_max   = float(m_max)
        self.m_min   = float(m_min)
        self.n_max   = float(n_max)
        self.S_max   = float(S_max)
        self.kappa   = n_max / S_max
        self.se_hat  = se_phys / S_max
        self.ks_max  = ks_max
        self.K_s     = K_s

    def forward(self, t, x):
        """RHS for normalized state x = [n_hat, S_hat]"""
        n_hat = x[:-1]
        S_hat = x[-1]

        # alpha(S)
        S_phys = self.S_max * S_hat
        alpha  = self.ks_max * S_phys / (self.K_s + S_phys + 1e-12)

        # transport
        rg = alpha * self.m
        rg_n = rg * n_hat
        flux_out = rg_n[-1]
        rg_n = rg_n.clone()
        rg_n[-1] = 0.0

        d_rg_n_dm = torch.zeros_like(n_hat)
        d_rg_n_dm[1:] = (rg_n[1:] - rg_n[:-1]) / self.delta_m
        d_rg_n_dm[0]  = rg_n[0] / self.delta_m
        T = -d_rg_n_dm
        T[-1] += flux_out / self.delta_m

        # division & dilution
        G = (1.0/(self.m_max - self.m + 1e-12)
             - 1.0/(self.m_max - self.m_min + 1e-12)).clamp(min=0.0) * rg
        division_loss = G * n_hat
        division_gain = 2.0 * self.delta_m * (G * n_hat) @ self.P
        dn_hat_dt = T - division_loss + division_gain

        # substrate
        uptake = torch.sum(rg * n_hat) * self.delta_m
        dS_hat_dt = - self.kappa * uptake

        return torch.cat([dn_hat_dt, dS_hat_dt.view(1)])

class PDEWrapper(nn.Module):
    def __init__(self, pde_model, delta_t):
        super().__init__()
        self.pde_model = pde_model
        self.delta_t = delta_t

    def forward(self, x, u=None):
        """
        x: (M+1,) normalized state [n_hat, S_hat]
        u: optional dilution rate D(t) (scalar)
        returns: x_next after delta_t
        """
        # current and next time
        t = torch.tensor([0.0, self.delta_t], dtype=x.dtype)

        # define RHS with optional dilution
        def rhs(t_, x_):
            return self.pde_model.forward(t_, x_) - (u if u is not None else 0.0) * torch.cat([x_[:-1], x_[-1:].view(1)])

        # integrate one step
        sol = odeint(rhs, x, t, method="dopri5")
        x_next = sol[-1]
        return x_next

