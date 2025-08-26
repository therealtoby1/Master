######################Neural Networks###########################################
import torch
import torch.nn as nn
import gpytorch

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
