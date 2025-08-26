#file for functions that help training the NN models and make the code more readable
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import plotly.graph_objects as go
import numpy as np






########################################NEURAL NETWORK TRAINING FUNCTIONS########################################
#ADAM OPTIMIZER ON FULL DATASET (SMALL DATASETS); large datasets should use SGD as my laptop crashed twice

def train_the_model_full_dataset_Adam(model, X_train, Y_train, num_epochs=300,loss_function=None,**kwargs):

    " A function to help train the model usinng the ADAM optimizer on the whole dataset (used for the batch optimization)"
    "kwargs may be added and will be used in a Loss function that the user can also define"



    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    if loss_function is not None:
        criterion = loss_function
    else : 
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
        loss = criterion(output, Y_train, **kwargs)
        loss.backward()
        optimizer.step()

        scheduler.step(loss.item())

        # --- save best weights ---
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 100 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Iter {epoch:05d} | Loss: {loss.item():.8f} | Best: {best_loss:.8f} | LR: {lr:.2e}")
        epoch += 1

        # # optional break condition
        # if best_loss <= 1e-4:
        #     break

    # --- load best weights into model ---
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_loss

def PINN_loss_fun_batch(output, Y_train, *, n_max, s_max, mass_grid, lamb=1e-3, w_nonneg=1e2):
    """A custom loss function that can include the preservation of mass for the integro differential equation. 
    It can be directly used in the training function defined above"""
    #denormalize for the physics constraint!

    mse=nn.MSELoss()
    n_pred_denorm=output[:, :-1] * n_max 
    s_pred_denorm=output[:, -1] * s_max 

    n_Y_train_denorm = Y_train[:, :-1] * n_max
    s_Y_train_denorm = Y_train[:, -1] * s_max

    entire_initial_mass=(n_Y_train_denorm * mass_grid).sum(dim=1)*(mass_grid[1]-mass_grid[0]) + s_Y_train_denorm
    # biomass + substrate for each batch element
    K      = entire_initial_mass
    K_pred = (n_pred_denorm  * mass_grid).sum(dim=1) * (mass_grid[1]-mass_grid[0]) + s_pred_denorm

    # physics loss + data loss
    loss_data   = mse(output, Y_train)
    loss_phys   = mse(K, K_pred)

    # substrate nonnegativity penalty
    S_pred = s_pred_denorm
    loss_nonneg = w_nonneg * torch.mean(torch.relu(-S_pred) ** 2)

    # total loss
    loss = loss_data +lamb*loss_phys
    return loss

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


#######################################################DYNAMIC MODE DECOMPOSITION########################################################

def EDMD_feature_space(X, degree=3):
    """
    FUNCTION FOR EDMD to create the feature space for the DMD algorithm
    ------------------------------------------------------------------
    Build feature matrix with:
      - distribution n (linear)
      - substrate polynomials up to 'degree'
      - cross terms n * S^k
      - RBF features in substrate S

    Args:
      X: (N, M+1) snapshots, rows = [n_vec, S]
      degree: polynomial degree for substrate (default=2)
    Returns:
      Phi: (N, feature_dim)
    """
    N, d = X.shape
    M = d - 1
    n = X[:, :M]          # (N, M)
    S = X[:, -1:]         # (N, 1)
    eps=1e-8
    features = [n]  # always keep full distribution

    # substrate polynomials: S, S^2, ..., S^degree
    for k in range(1, degree+1):
        Sk = S**k
        features.append(Sk)
        features.append((Sk+n))
        features.append(n* Sk)  # cross terms with n
    

 
    Phi = torch.cat(features, dim=1)
    return Phi









































#######################################################Plotting Functions########################################################

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