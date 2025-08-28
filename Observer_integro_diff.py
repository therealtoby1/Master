# ============================================================
# Unscented Kalman Filter with Substrate as Last State
# The Filter works with already normalized inputs and measurments
# ============================================================
import warnings
import torch
import matplotlib.pyplot as plt
import numpy as np
import gpytorch

class UKF:
    def __init__(self, state_dim, meas_dim, f_model, h_model, n_max, s_max,
                 Q=None, R=None, P0=None, alpha=0.1, beta=2.0, kappa=0.0,m_min=0):
        self.n = state_dim
        self.m = meas_dim
        self.f_model = f_model
        self.h_model = h_model
        self.n_max = n_max
        self.s_max = s_max
        self.m_min = m_min

        self.Q = Q if Q is not None else torch.eye(self.n) * 1e-3
        self.R = R if R is not None else torch.eye(self.m) * 1e-2
        self.P = P0 if P0 is not None else torch.eye(self.n)*1e-2
        self.x = torch.zeros(self.n)      #initial guess- pls assign value

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n

        self.Wm = torch.full((2*self.n+1,), 0.5/(self.n + self.lambda_))
        self.Wc = self.Wm.clone()
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

    # --------------------------------------------------------
    def set_initial_state(self,x_phys):
        x_norm=x_phys.clone()
        x_norm[:-1] = x_norm[:-1] / self.n_max
        x_norm[-1]  = x_norm[-1]  / self.s_max
        self.x = x_norm

    def sigma_points(self, x, P):
        # ensure symmetry
        P = 0.5 * (P + P.T)
        S = (self.n + self.lambda_) * P

        # eigen-based square root (PSD-safe)
        evals, evecs = torch.linalg.eigh(S)
        eps = 1e-12
        evals = torch.clamp(evals, min=eps)
        # U has columns = spread directions (n x n)
        U = evecs @ torch.diag(torch.sqrt(evals))

        # build sigma points using COLUMNS of U
        sigmas = [x]
        for i in range(self.n):
            col = U[:, i]
            sigmas.append(x + col)
            sigmas.append(x - col)
        return torch.stack(sigmas)

    # --------------------------------------------------------
    def unscented_transform(self, sigmas, noise_cov=None):
        mean = torch.sum(self.Wm[:, None] * sigmas, dim=0)
        diffs = sigmas - mean
        d = sigmas.shape[1]
        cov = torch.zeros((d, d))
        for i in range(sigmas.size(0)):
            cov += self.Wc[i] * torch.outer(diffs[i], diffs[i])
        if noise_cov is not None:
            cov += noise_cov
        return mean, cov

    # --------------------------------------------------------
    def step(self, u, y_meas):
        """Perform one UKF predict + update step."""
        sigmas = self.sigma_points(self.x, self.P)
        sigmas_pred = torch.stack([self.f_model(s, u) for s in sigmas])

         
        x_pred, P_pred = self.unscented_transform(sigmas_pred, noise_cov=self.Q)

        sigmas_meas = torch.stack([self.h_model(s) for s in sigmas_pred])
        y_pred, Pyy = self.unscented_transform(sigmas_meas, noise_cov=self.R)
        
        Pxy = torch.zeros((self.n, self.m))
        for i in range(sigmas.size(0)):
            dx = sigmas_pred[i] - x_pred
            dy = sigmas_meas[i] - y_pred
            Pxy += self.Wc[i] * dx[:, None] @ dy[None, :]

        K = Pxy @ torch.linalg.inv(Pyy)
        # if self.m_min ==0:
        #     K[0,:]= 0.0


        self.x = x_pred + (K @ (y_meas - y_pred)).squeeze()
        self.P = P_pred - K @ Pyy @ K.T
        # after computing self.P in step()
        self.P = 0.5 * (self.P + self.P.T)               # symmetrize
        evals, evecs = torch.linalg.eigh(self.P)
        self.P = (evecs * torch.clamp(evals, min=1e-12)) @ evecs.T

        return self.x, self.P

    # --------------------------------------------------------
    def run(self, U_seq, Y_seq):
        """
        Run the UKF for a sequence of timesteps.
        Returns X_estimates of shape (T+1, n) including the initial guess at index 0.
        """
        T = len(Y_seq)
        X_estimates = torch.zeros((T+1, self.n), dtype=self.x.dtype, device=self.x.device)

        # store initial guess before any correction
        X_estimates[0] = self.x

        #check if normalization happened earlier
        if (X_estimates > 10).any():
            warnings.warn(
                "Suspiciously large values in initial state. "
                "Did you forget to normalize? "
                "Use set_initial_state() for automatic normalization."
            )

        for t in range(T):
            u = U_seq[t] if U_seq is not None else None
            y_meas = Y_seq[t]
            x_new, _ = self.step(u, y_meas)
            X_estimates[t+1] = x_new
            print(X_estimates[t+1,50])


        
        X_phys = X_estimates.clone()
        X_phys[:, :-1] *= self.n_max
        X_phys[:, -1]  *= self.s_max
        return X_phys
    #------------------------------------------------------------
    def plot_results(self, X_estimates, biomass_measurement,
                    S_reference=None, m_torch=None, delta_m_torch=None,
                    title_prefix="UKF"):
        """
        Plot biomass (measured vs estimated) and substrate (estimated vs reference)
        in one figure with 2 subplots.
        """
        # ensure numpy arrays
        if torch.is_tensor(biomass_measurement):
            bm_meas_np = biomass_measurement.detach().cpu().numpy()
        else:
            bm_meas_np = np.array(biomass_measurement)

        # extract estimated distribution + substrate
        n_est_traj = X_estimates[:, :-1]
        S_est_traj = X_estimates[:, -1]

        # compute biomass from estimated distribution
        if m_torch is None or delta_m_torch is None:
            raise ValueError("m_torch and delta_m_torch are required to compute biomass.")
        biomass_est = (n_est_traj * m_torch).sum(dim=1) * delta_m_torch
        biomass_est_np = biomass_est.detach().numpy()
        S_est_np = S_est_traj.detach().numpy()

        # substrate reference if available
        S_ref_np = None
        if S_reference is not None:
            if torch.is_tensor(S_reference):
                S_ref_np = S_reference.detach().numpy()
            else:
                S_ref_np = np.array(S_reference)

        # --- Plot ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # Biomass subplot
        ax1.plot(bm_meas_np, '-', label='Measured biomass')
        ax1.plot(biomass_est_np, 'o', markersize=2, label=f'{title_prefix} biomass estimate')
        ax1.set_ylabel("Biomass")
        ax1.grid(True); ax1.legend()

        # Substrate subplot
        if S_ref_np is not None:
            ax2.plot(S_ref_np, '-', label="Reference substrate")
        ax2.plot(S_est_np, 'o', markersize=2, label=f'{title_prefix} substrate estimate')
        
        ax2.set_xlabel("Timestep"); ax2.set_ylabel("Substrate")
        ax2.grid(True); ax2.legend()

        plt.tight_layout()
        plt.show()


    #----------------------------------------------------evaluating error function
    def evaluate_errors(self, X_estimates, biomass_true, S_true, 
                    m_torch, delta_m_torch,
                    rel_tol=0.05, abs_tol=1e-6, window=10):
        """
        Evaluate UKF errors for biomass + substrate against reference trajectories.
        Ignores the first state (initial guess) and the last state (no measurement).
        """
        # slice states to match measurements
        X = X_estimates[1:-1].clone()

        # slice reference trajectories (skip initial value, align lengths)
        biomass_true = torch.as_tensor(biomass_true, dtype=torch.float32).view(-1)[1:]
        S_true = torch.as_tensor(S_true, dtype=torch.float32).view(-1)[1:]


        # estimated outputs
        biomass_est = (X[:, :-1] * m_torch).sum(dim=1) * delta_m_torch
        substrate_est = X[:, -1]

        # compute metrics
        biomass_metrics = error_metrics(biomass_true, biomass_est,
                                        rel_tol=rel_tol, abs_tol=abs_tol, window=window)
        substrate_metrics = error_metrics(S_true, substrate_est,
                                        rel_tol=rel_tol, abs_tol=abs_tol, window=window)

        return {
            "biomass": biomass_metrics,
            "substrate": substrate_metrics
        }
    



#-------------------------------------------------------END OF UKF--------------------------------------------------------
def error_metrics(y_true, y_est, rel_tol=0.05, abs_tol=1e-6, window=10):
    """
    Compute error metrics between true and estimated trajectories.
    
    Args:
        y_true: (T,) ground truth trajectory
        y_est:  (T,) estimated trajectory
        rel_tol: relative tolerance (fraction of mean(|y_true|))
        abs_tol: absolute tolerance floor
        window: consecutive steps required for settling

    Returns:
        dict with RMSE, MAE, relative norms, time_in_tol, settling_time
    """
    y_true = torch.as_tensor(y_true, dtype=torch.float32).view(-1)
    y_est  = torch.as_tensor(y_est, dtype=torch.float32).view(-1)
    e = y_est - y_true
    T = len(y_true)

    # dynamic tolerance
    tol_val = max(abs_tol, rel_tol * y_true.abs().mean().item())

    rmse = torch.sqrt((e**2).mean()).item()
    mae  = e.abs().mean().item()
    max_err = e.abs().max().item()
    rel_l2 = torch.norm(e, p=2).item() / (torch.norm(y_true, p=2).item() + 1e-12)

    # percentage of time within tol
    time_in_tol = (e.abs() < tol_val).float().mean().item()

    # settling time (first index where error stays < tol for 'window' steps)
    settling_time = T
    for t in range(T - window):
        if (e[t:t+window].abs() < tol_val).all():
            settling_time = t
            break

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MaxError": max_err,
        "Rel_L2": rel_l2,
        "time_in_tol (%)": time_in_tol,
        "settling_time(Timesteps)": settling_time,
        "tolerance_used": tol_val,
    }

def rollout(model, n_norm0, s_norm0, steps):
    """
    model: maps [n_norm(t), s_norm(t)] -> [n_norm(t+1), s_norm(t+1)]
    n_norm0: (M,) torch tensor, already normalized
    s_norm0: scalar or tensor, already normalized
    """
    model.eval()
    n = n_norm0.float().view(-1)
    s = s_norm0.float().view(())
    M = n.shape[0]

    N = torch.empty((M, steps+1))
    S = torch.empty((steps+1,))
    N[:,0] = n; S[0] = s

    with torch.no_grad():
        for t in range(steps):
            x_norm = torch.cat([N[:,t], S[t].unsqueeze(0)], dim=0)  # already normalized
            y_norm = model(x_norm.unsqueeze(0)).squeeze(0)          # (M+1,)
            n_next = y_norm[:M]
            s_next = y_norm[M]
            N[:,t+1] = n_next
            S[t+1] = s_next
    return N, S


#f_model for one forward step using the rollout function that has already been defined: 

def f_model_NN(x, model):
    """
    One-step dynamics wrapper for UKF using rollout().
    
    x: state vector (M+1,) = [distribution, substrate]
    model: your trained NN model
    n_max: normalization factor for distribution
    S_max: max substrate
    
    Returns: x_next (M+1,) torch tensor
    """
    M = x.shape[0] - 1
    n0 = x[:-1]
    s0 = x[-1]

    # rollout one step
    N, S = rollout(model, n_norm0=n0, s_norm0=s0,steps=1)

    # grab the next step
    n_next = N[:, 1]
    s_next = S[1]
    return torch.cat([n_next, s_next.view(1)], dim=0)

def f_model_GP(x, model, likelihood):
    """
    One-step dynamics wrapper for UKF using a trained GP.
    
    Args:
        x: state vector (M+1,) normalized
        model: GPyTorch model
        likelihood: GPyTorch likelihood
    
    Returns:
        mean_next: (M+1,) torch tensor (normalized)
    """
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # GP expects batch dimension
        x_in = x.unsqueeze(0)   # (1, D)
        
        # predictive distribution
        pred = likelihood(model(x_in))    # MultitaskMultivariateNormal
        mean = pred.mean.squeeze(0)       # (M+1,)
        
    return mean


####f_model_DMD######
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

def f_model_EDMD(x,
                U_r,            #matrix for reduced order approx
                K_tilde,        #forward matrix for phi_k+1 = K_tilde @ phi_k
                degree):        #degree of feature space polynomial
    dim=x.size(0)
    phi0 = EDMD_feature_space(x.unsqueeze(0),degree=degree).squeeze(0)  # (2M+1,)
    z0 = U_r.T @ phi0
    z_pred = K_tilde @ z0
    Phi_pred = (U_r @ z_pred.T).T  
    x_est=Phi_pred[:dim]

    return x_est

def NN_feature_space(X, degree=3):
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
        # features.append(n**k)
        features.append((Sk+n))
        features.append(n* Sk)  # cross terms with n
    
    #we end up with M+ degree*(201)
 
    Phi = torch.cat(features, dim=1)
    return Phi

def f_model_NN_feature_space(model, x, degree, steps=1):
    """
    One-step or multi-step rollout for a NN trained in feature space.
    Assumes x = [n (M,), s] is ALREADY normalized (no scaling inside).
    
    Args:
        model: trained NN (input = EDMD features of normalized state)
        x:     (M+1,) torch tensor, normalized state [n, s]
        degree: polynomial degree for EDMD_feature_space
        steps: number of rollout steps to apply

    Returns:
        x_next: (M+1,) torch tensor, normalized state after 'steps'
    """
    model.eval()

    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.float().view(-1)

    M = x.shape[0] - 1  # number of bins (last entry is substrate)

    with torch.no_grad():
        x_curr = x
        for _ in range(steps):
            # lift to feature space
            phi_x = NN_feature_space(x_curr.unsqueeze(0), degree=degree)  # (1, feature_dim)

            # predict next feature vector
            y_norm = model(phi_x).squeeze(0)  # (feature_dim,)

            # extract next normalized state (assume first M+1 are raw state)
            n_next = y_norm[:M]
            s_next = y_norm[M]
            x_curr = torch.cat([n_next, s_next.view(1)], dim=0)

    return x_curr
