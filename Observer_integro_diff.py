# ============================================================
# Unscented Kalman Filter with Substrate as Last State
# ============================================================
import torch
class UKF:
    def __init__(self, state_dim, meas_dim, f_model, h_model, 
                 Q=None, R=None, P0=None, alpha=0.1, beta=2.0, kappa=0.0):
        self.n = state_dim
        self.m = meas_dim
        self.f_model = f_model
        self.h_model = h_model

        self.Q = Q if Q is not None else torch.eye(self.n) * 1e-3
        self.R = R if R is not None else torch.eye(self.m) * 1e-2
        self.P = P0 if P0 is not None else torch.eye(self.n)*1e-2
        self.x = torch.zeros(self.n)

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n

        self.Wm = torch.full((2*self.n+1,), 0.5/(self.n + self.lambda_))
        self.Wc = self.Wm.clone()
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

    # --------------------------------------------------------
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

        for t in range(T):
            u = U_seq[t] if U_seq is not None else None
            y_meas = Y_seq[t]
            x_new, _ = self.step(u, y_meas)
            X_estimates[t+1] = x_new

        return X_estimates


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

def f_model(x, model):
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