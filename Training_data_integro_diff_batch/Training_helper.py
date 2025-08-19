#file for functions that help training the NN models and make the code more readable
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import plotly.graph_objects as go
import numpy as np







#ADAM OPTIMIZER ON FULL DATASET (SMALL DATASETS); large datasets should use SGD as my laptop crashed twice

def train_the_model_full_dataset_Adam(model, X_train, Y_train, num_epochs=300):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20,
        threshold=1e-6,
        min_lr=1e-6,
    )

    best_loss = float('inf')
    best_state = None

    epoch = 0
    while epoch < num_epochs:
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, Y_train)
        loss.backward()
        optimizer.step()

        scheduler.step(loss.item())

        # --- save best weights ---
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 100 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Iter {epoch:05d} | Loss: {loss.item():.6f} | Best: {best_loss:.6f} | LR: {lr:.2e}")
        epoch += 1

        # # optional break condition
        # if best_loss <= 1e-4:
        #     break

    # --- load best weights into model ---
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_loss

#run a simulation
def rollout(model, n0, s0, steps, n_max, S_max=5.0, return_numpy=True):
    """
    model: maps [n_norm(t), s_norm(t)] -> [n_norm(t+1), s_norm(t+1)]
    n0:    (M,) torch tensor, unnormalized
    s0:    scalar or tensor, unnormalized
    steps: number of steps
    n_max: global max for log normalization
    S_max: maximum substrate (default 5.0)

    Returns:
        N: (M, steps+1) tensor with unnormalized n(0..steps)
        S: (steps+1,) tensor with unnormalized s(0..steps)
    """
    model.eval()
    if not torch.is_tensor(n0):
        n0 = torch.tensor(n0, dtype=torch.float32)
    n = n0.float().view(-1)
    s = torch.as_tensor(s0, dtype=torch.float32).view(())

    M = n.shape[0]
    N = torch.empty((M, steps + 1), dtype=torch.float32)
    S = torch.empty((steps + 1,), dtype=torch.float32)
    N[:, 0] = n
    S[0] = s

    if not torch.is_tensor(n_max):
        log_n_max = torch.log1p(torch.tensor(n_max, dtype=torch.float32))
    else: 
        log_n_max = torch.log1p(n_max)

    with torch.no_grad():
        for t in range(steps):
            # --- normalize current state ---
            n_norm = N[:,t]/n_max #torch.log1p(N[:, t]) / log_n_max
            s_norm = S[t] / S_max
            x_norm = torch.cat([n_norm, s_norm.unsqueeze(0)], dim=0)  # (M+1,)

            # --- predict next state in normalized space ---
            y_norm = model(x_norm.unsqueeze(0)).squeeze(0)  # (M+1,)

            # --- invert normalization ---
            # n_next = torch.expm1(y_norm[:M] * log_n_max)
            n_next = y_norm[:M]*n_max
            s_next = y_norm[M] * S_max

            # --- store ---
            N[:, t + 1] = n_next
            S[t + 1] = s_next

    if return_numpy:
        return N.numpy(), S.numpy()
    return N, S


def plot_spaghetti(m, t, N, title="", step=2, colorscale='Viridis'):
    """
    Create a 3D spaghetti plot for n(m, t).
    
    Parameters
    ----------
    m : 1D array
        Mass values.
    t : 1D array
        Time values.
    N : 2D array, shape (len(m), len(t))
        n(m, t) data.
    title : str
        Plot title.
    step : int
        Time index step for subsampling curves.
    colorscale : str
        Plotly colorscale name.
    """
    # Meshgrid
    T, M_ = np.meshgrid(t, m)
    
    # Shared color range
    cmin, cmax = N.min(), N.max()
    
    # Subsample time slices
    time_idxs = np.arange(0, N.shape[1], step)
    
    traces = []
    for i, k in enumerate(time_idxs):
        traces.append(
            go.Scatter3d(
                x=M_[:, k],
                y=T[:, k],
                z=N[:, k],
                mode='lines',
                showlegend=False,
                line=dict(
                    color=N[:, k],
                    colorscale=colorscale,
                    cmin=cmin,
                    cmax=cmax,
                    width=4,
                    showscale=(i == 0),
                    colorbar=dict(title='n(m, t)') if i == 0 else None
                )
            )
        )

    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='Mass (m)',
                       autorange="reversed"),
            yaxis=dict(title='Time (t)'),
            zaxis=dict(title='n(m, t)')
        ),
        width=700,
        height=700
    )
    fig.show()


def shifted_gaussian(m, mu, sigma):
    """
    Gaussian-like curve on [0, m_max] with value 0 at m=0,
    normalized so that biomass = 1.

    Parameters:
        m      : torch tensor of domain values (e.g. torch.linspace)
        mu     : mean (float or torch scalar)
        sigma  : std deviation (float or torch scalar)
        delta_m: grid spacing (float)

    Returns:
        torch tensor same shape as m
    """
    # Standard Gaussian
    g = torch.exp(-0.5 * ((m - mu) / sigma) ** 2)

    # Subtract value at m=0
    g0 = torch.exp(-0.5 * ((0.0 - mu) / sigma) ** 2)
    g_shifted = g - g0

    # Clip negatives (like np.maximum)
    g_shifted = torch.clamp(g_shifted, min=0.0)

    # Normalize: biomass = ∑ m * g_shifted * delta_m = 1
    total = torch.sum(m * g_shifted) * (m[1]-m[0])
    g_shifted = g_shifted / total

    return g_shifted