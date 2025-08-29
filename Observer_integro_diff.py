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


        
        X_phys = X_estimates.clone()
        X_phys[:, :-1] *= self.n_max
        X_phys[:, -1]  *= self.s_max
        return X_phys
    #------------------------------------------------------------plotting----------------------
    def plot_results(self, X_estimates, biomass_measurement, 
                 S_reference=None, m_torch=None, delta_m_torch=None,
                 title_prefix="UKF",
                 rel_tol=0.01, abs_tol=0.05,
                 y_max_mass=5, y_max_substrate=5,
                 window=10,
                 plot_tolerance=True,
                 show_x0=True):
        """
        Plot biomass (measured vs estimated) and substrate (estimated vs reference)
        in one figure with 2 subplots.

        - Uses error_metrics to draw tolerance bands (if plot_tolerance=True).
        - Aligns estimates as *post-update*: y_k (t=k) -> x_{k+1} (t=k+1).
        That is, if X_estimates has length T+1 (x0 + x1..xT) and measurements have length T,
        we plot estimates at t=1..T, and show x0 as a marker at t=0.
        """

        import numpy as np
        import matplotlib.pyplot as plt
        import torch

        # --- to tensors (CPU for plotting) ---
        bm_meas = biomass_measurement.detach().cpu() if torch.is_tensor(biomass_measurement) \
                else torch.as_tensor(biomass_measurement, dtype=torch.float32)
        if not torch.is_tensor(X_estimates):
            X_estimates = torch.as_tensor(X_estimates, dtype=torch.float32)

        S_ref = None
        if S_reference is not None:
            S_ref = S_reference.detach().cpu() if torch.is_tensor(S_reference) \
                    else torch.as_tensor(S_reference, dtype=torch.float32)

        # --- lengths ---
        Tm = bm_meas.shape[0]          # number of measurements
        Te = X_estimates.shape[0]      # number of state entries (often Tm+1)

        # --- alignment & time indices ---
        # Case A (typical): Te >= Tm + 1 -> we have x0 and posteriors x1..xTm
        if Te >= Tm + 1:
            L = Tm
            X = X_estimates[1:1+L].clone()     # x1..xL
            t_meas = np.arange(L)              # 0..L-1 (y_0..y_{L-1})
            t_est  = np.arange(1, L+1)         # 1..L   (x_1..x_L)
            x0_state = X_estimates[0].clone() if show_x0 else None
            # trim references to L if present
            if S_ref is not None:
                S_ref = S_ref[:L]
        # Case B: Te == Tm -> no separate x0 row; use same-index alignment safely
        elif Te == Tm:
            L = Tm
            X = X_estimates[:L].clone()
            t_meas = np.arange(L)              # 0..L-1
            t_est  = np.arange(L)              # 0..L-1 (x_k aligned to y_k)
            x0_state = X_estimates[0].clone() if (show_x0 and Te > 0) else None
            if S_ref is not None:
                S_ref = S_ref[:L]
        # Case C: Te < Tm -> shorten measurements to Te (best-effort)
        else:
            L = Te
            X = X_estimates[:L].clone()
            bm_meas = bm_meas[:L]
            t_meas = np.arange(L)
            t_est  = np.arange(L)
            x0_state = X_estimates[0].clone() if (show_x0 and Te > 0) else None
            if S_ref is not None:
                S_ref = S_ref[:L]

        # --- split state ---
        n_est_traj = X[:, :-1]
        S_est_traj = X[:, -1]

        # --- biomass estimate (Σ n_i m_i Δm) ---
        if m_torch is None or delta_m_torch is None:
            raise ValueError("m_torch and delta_m_torch are required to compute biomass.")

        m_vec = m_torch.reshape(-1) if torch.is_tensor(m_torch) \
                else torch.as_tensor(m_torch, dtype=torch.float32).reshape(-1)
        delta_m = (delta_m_torch.reshape(()).to(torch.float32) if torch.is_tensor(delta_m_torch)
                else torch.tensor(float(delta_m_torch), dtype=torch.float32))

        biomass_est = (n_est_traj * m_vec).sum(dim=1) * delta_m
        biomass_est_np = biomass_est.detach().cpu().numpy()
        S_est_np = S_est_traj.detach().cpu().numpy()

        # --- x0 markers (only for plotting; does not affect metrics/bands) ---
        x0_biomass = None
        x0_substrate = None
        if show_x0 and (x0_state is not None):
            x0_biomass = float((x0_state[:-1] * m_vec).sum() * delta_m)
            x0_substrate = float(x0_state[-1])

        # --- tolerance bands via error_metrics (use aligned sequences of length L) ---
        biomass_tol = None
        substrate_tol = None
        if plot_tolerance:
            bm_metrics = error_metrics(
                bm_meas[:L], biomass_est,
                rel_tol=rel_tol, abs_tol=abs_tol, window=window,
                y_max=y_max_mass
            )
            biomass_tol = bm_metrics["tolerance_used (per step)"]

            if S_ref is not None:
                sub_metrics = error_metrics(
                    S_ref, S_est_traj,
                    rel_tol=rel_tol, abs_tol=abs_tol, window=window,
                    y_max=y_max_substrate
                )
                substrate_tol = sub_metrics["tolerance_used (per step)"]

        # --- numpy views for plotting ---
        bm_meas_np = bm_meas[:L].detach().cpu().numpy()
        S_ref_np = S_ref.detach().cpu().numpy() if S_ref is not None else None

        # --- Plot ---
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # Biomass subplot
        if plot_tolerance and (biomass_tol is not None):
            ax1.fill_between(
                t_meas, bm_meas_np - biomass_tol, bm_meas_np + biomass_tol,
                color='gray', alpha=0.3, label="Tolerance band"
            )
        ax1.plot(t_meas, bm_meas_np, '-', label='Measured biomass')
        ax1.plot(t_est, biomass_est_np, 'o', markersize=2, label=f'{title_prefix} biomass est')

        # x0 at t=0
        if x0_biomass is not None and len(t_meas) > 0:
            ax1.plot([0], [x0_biomass], marker='s', markersize=5,
                    linestyle='None', label='x0 (estimate)')

        ax1.set_ylabel("Biomass")
        ax1.grid(True); ax1.legend()

        # Substrate subplot
        if S_ref_np is not None:
            if plot_tolerance and (substrate_tol is not None):
                ax2.fill_between(
                    t_meas, S_ref_np - substrate_tol, S_ref_np + substrate_tol,
                    color='gray', alpha=0.3, label="Tolerance band"
                )
            ax2.plot(t_meas, S_ref_np, '-', label="Reference substrate")

        ax2.plot(t_est, S_est_np, 'o', markersize=2, label=f'{title_prefix} substrate est')

        # x0 substrate at t=0
        if x0_substrate is not None and len(t_meas) > 0:
            ax2.plot([0], [x0_substrate], marker='s', markersize=5,
                    linestyle='None', label='x0 (estimate)')

        ax2.set_xlabel("Timestep"); ax2.set_ylabel("Substrate")
        ax2.grid(True); ax2.legend()

        plt.tight_layout()
        plt.show()
     

        

        



    #----------------------------------------------------evaluating error function
    def evaluate_errors(self, X_estimates, biomass_true, S_true, 
                    m_torch, delta_m_torch,
                    rel_tol=0.05, abs_tol=1e-6, window=10,y_max_mass=5,y_max_substrate=5):
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
                                        rel_tol=rel_tol, abs_tol=abs_tol, window=window,y_max=y_max_mass)
        substrate_metrics = error_metrics(S_true, substrate_est,
                                        rel_tol=rel_tol, abs_tol=abs_tol, window=window,y_max=y_max_substrate)

        return {
            "biomass": biomass_metrics,
            "substrate": substrate_metrics
        }
    


#-------------------------------------------------------START OF EKF------------------------------------------------------



import torch

class EKF:
    def __init__(self, state_dim, meas_dim, f_model, h_model,
                 n_max=1.0, s_max=1.0, use_autograd=True,
                 F_linearize_at="posterior"):
        """
        EKF with normalized internal state; returns physical estimates (tensor).

        State: x = [n_1,...,n_M, S], state_dim = M+1 (S last).
        Internally: x_norm = x_phys / scale, scale = [n_max (M copies), s_max].
        f_model, h_model take normalized x; h_model returns PHYSICAL measurement.
        Q in normalized units, R in physical units.

        F_linearize_at in {"posterior","last_pred"} controls where to linearize F.
        """
        assert F_linearize_at in ("posterior", "last_pred")
        self.n = state_dim
        self.m = meas_dim
        self.f_model = f_model
        self.h_model = h_model
        self.use_autograd = use_autograd
        self.F_linearize_at = F_linearize_at

        # filter state (normalized)
        self.x = torch.zeros(self.n, dtype=torch.float32)
        self.P = torch.eye(self.n, dtype=torch.float32)

        #initial condition (not normalized)
        self.x0_phys = None        

        # covariances
        self.Q = torch.eye(self.n, dtype=torch.float32) * 1e-3   # normalized space
        self.R = torch.eye(self.m, dtype=torch.float32) * 1e-2   # physical space

        # build scaling vector from SCALARS n_max, s_max
        M = self.n - 1
        n_scalar = self._as_scalar(n_max, "n_max")
        s_scalar = self._as_scalar(s_max, "s_max")
        n_scale = torch.full((M,), n_scalar, dtype=torch.float32)
        s_scale = torch.tensor([s_scalar], dtype=torch.float32)
        self.scale = torch.cat([n_scale, s_scale])  # (n,)

        # store last a-priori for optional linearization policy
        self._x_last_pred = None

    # ---------- helpers ----------
    @staticmethod
    def _as_scalar(val, name="value"):
        t = torch.as_tensor(val, dtype=torch.float32).reshape(-1)
        assert t.numel() == 1, f"{name} must be a single scalar."
        return float(t.item())

    def _to_norm(self, x_phys):
        x = torch.as_tensor(x_phys, dtype=torch.float32).reshape(-1)
        return x / self.scale

    def _to_phys(self, x_norm):
        return x_norm * self.scale

    def set_initial_state(self, x0_phys):
        self.x0_phys= x0_phys
        self.x = self._to_norm(x0_phys)
        self._x_last_pred = None

    # ---------- Jacobian ----------
    def _jacobian(self, func, x_norm, eps=1e-5):
        if self.use_autograd:
            try:
                x = x_norm.detach().clone().requires_grad_(True)
                y = func(x).reshape(-1)
                k = y.numel()
                J = torch.zeros(k, x.numel(), dtype=torch.float32)
                for i in range(k):
                    (gi,) = torch.autograd.grad(y[i], x, retain_graph=True, allow_unused=True)
                    if gi is None:
                        gi = torch.zeros_like(x)
                    J[i] = gi
                return J
            except Exception:
                pass  # fallback to FD

        # finite differences
        x = x_norm.detach()
        y0 = func(x).reshape(-1)
        k = y0.numel()
        J = torch.zeros(k, x.numel(), dtype=torch.float32)
        for i in range(x.numel()):
            dx = torch.zeros_like(x)
            dx[i] = eps
            yi = func(x + dx).reshape(-1)
            J[:, i] = (yi - y0) / eps
        return J

    # ---------- EKF steps (normalized space) ----------
    def predict(self, u=None):
        # Choose linearization point for F
        if self.F_linearize_at == "last_pred" and (self._x_last_pred is not None):
            x_lin = self._x_last_pred
        else:
            x_lin = self.x  # standard: last posterior

        F = self._jacobian(lambda xx: self.f_model(xx, u), x_lin)

        # Propagate mean & covariance
        x_pred = self.f_model(self.x, u).reshape(-1)
        PHt = self.P @ F.T
        P_pred = F @ PHt + self.Q

        self.x = x_pred
        self.P = P_pred
        self._x_last_pred = x_pred.detach()

    def update(self, y_phys):
        # Predict measurement (PHYSICAL units) at a-priori state
        y_pred = self.h_model(self.x).reshape(-1)
        H = self._jacobian(self.h_model, self.x)

        y = torch.as_tensor(y_phys, dtype=torch.float32).reshape(-1)
        innov = y - y_pred

        S = H @ self.P @ H.T + self.R
        PHt = self.P @ H.T
        # Solve S K^T = (P H^T)^T  => K = (S \ PHt^T)^T (more stable than explicit inverse)
        K = torch.linalg.solve(S, PHt.T).T

        self.x = self.x + K @ innov
        I = torch.eye(self.n, dtype=torch.float32)
        self.P = (I - K @ H) @ self.P

    def run(self, U_seq, Y_seq):
        """
        U_seq: (T, ?) or None
        Y_seq: (T, meas_dim) in PHYSICAL units (torch tensor or array-like)
        Returns:
            X_phys: (T, state_dim) torch.Tensor in PHYSICAL units
        """
        T = Y_seq.shape[0]
        X_phys = torch.zeros(T + 1, self.n, dtype=torch.float32)
        X_phys[0] = self.x0_phys.clone()
        for t in range(T):
            u_t = None if U_seq is None else U_seq[t]
            self.predict(u_t)
            self.update(Y_seq[t])
            X_phys[t+1] = self._to_phys(self.x)
        return X_phys.detach()

    def plot_results(self, X_estimates, biomass_measurement,
                    S_reference=None, m_torch=None, delta_m_torch=None,
                    title_prefix="EKF"):
        import numpy as np
        import matplotlib.pyplot as plt
        import torch

        if torch.is_tensor(biomass_measurement):
            bm_meas_np = biomass_measurement.detach().cpu().numpy()
        else:
            bm_meas_np = np.array(biomass_measurement)

        # Ensure torch tensor for state trajectory (PHYSICAL units)
        if not torch.is_tensor(X_estimates):
            X_estimates = torch.as_tensor(X_estimates, dtype=torch.float32)

        n_est_traj = X_estimates[:, :-1]   # (T, M)
        S_est_traj = X_estimates[:, -1]    # (T,)

        if m_torch is None or delta_m_torch is None:
            raise ValueError("m_torch and delta_m_torch are required to compute biomass.")

        # ---- Coerce inputs permissively ----
        # m_torch: length-M vector (torch or array-like)
        if not torch.is_tensor(m_torch):
            m_vec = torch.as_tensor(m_torch, dtype=torch.float32).reshape(-1)
        else:
            m_vec = m_torch.reshape(-1)

        M = n_est_traj.shape[1]
        assert m_vec.numel() == M, f"m_torch must have length {M}."

        # delta_m_torch: allow python float / numpy scalar / torch scalar
        if torch.is_tensor(delta_m_torch):
            delta_m = delta_m_torch.reshape(()).to(torch.float32)
        else:
            delta_m = torch.tensor(float(delta_m_torch), dtype=torch.float32)

        # ---- Biomass estimate: sum_i n_i * m_i * Δm ----
        biomass_est = (n_est_traj * m_vec).sum(dim=1) * delta_m  # (T,)
        biomass_est_np = biomass_est.detach().numpy()
        S_est_np = S_est_traj.detach().numpy()

        # Substrate reference (optional)
        S_ref_np = None
        if S_reference is not None:
            if torch.is_tensor(S_reference):
                S_ref_np = S_reference.detach().numpy()
            else:
                S_ref_np = np.array(S_reference)

        # ---- Plot ----
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        ax1.plot(bm_meas_np, '-', label='Measured biomass')
        ax1.plot(biomass_est_np, 'o', markersize=2, label=f'{title_prefix} biomass estimate')
        ax1.set_ylabel("Biomass")
        ax1.grid(True); ax1.legend()

        if S_ref_np is not None:
            ax2.plot(S_ref_np, '-', label="Reference substrate")
        ax2.plot(S_est_np, 'o', markersize=2, label=f'{title_prefix} substrate estimate')

        ax2.set_xlabel("Timestep"); ax2.set_ylabel("Substrate")
        ax2.grid(True); ax2.legend()

        plt.tight_layout()
        plt.show()
        return fig, (ax1, ax2)


#The extended Kalman filter is another approach for a black box observer-->class of EKF, error metrics below should be able to be used, as well as the same f_functions and h_functions...































def error_metrics(y_true, y_est, rel_tol=0.05, abs_tol=1e-6, window=10, y_max=5):
    """
    Compute error metrics between true and estimated trajectories,
    with a dynamic tolerance band that shrinks with smaller y_true.

    Args:
        y_true: (T,) ground truth trajectory
        y_est:  (T,) estimated trajectory
        rel_tol: relative tolerance fraction
        abs_tol: absolute tolerance floor
        window: consecutive steps required for settling
        y_max: reference max value (scalar). If None, falls back to max(y_true).

    Returns:
        dict with RMSE, MAE, relative norms, time_in_tol, settling_time
    """
    y_true = torch.as_tensor(y_true, dtype=torch.float32).view(-1)
    y_est  = torch.as_tensor(y_est, dtype=torch.float32).view(-1)
    e = y_est - y_true
    T = len(y_true)

    # use external y_max if given, else fallback
    if y_max is None:
        y_max_val = y_true.abs().max().item()
    else:
        y_max_val = float(y_max)

    # --- dynamic tolerance per timestep ---
    tol_vec = torch.maximum(
        torch.full_like(y_true, abs_tol),
        rel_tol * y_max_val * y_true.abs()
    )

    rmse = torch.sqrt((e**2).mean()).item()
    mae  = e.abs().mean().item()
    max_err = e.abs().max().item()
    rel_l2 = torch.norm(e, p=2).item() / (torch.norm(y_true, p=2).item() + 1e-12)

    # percentage of time within tolerance band
    time_in_tol = (e.abs() < tol_vec).float().mean().item()

    # settling time (first index where error stays < tol for 'window' steps)
    settling_time = T
    for t in range(T - window):
        if (e[t:t+window].abs() < tol_vec[t:t+window]).all():
            settling_time = t
            break

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MaxError": max_err,
        "Rel_L2": rel_l2,
        "time_in_tol (%)": time_in_tol,
        "settling_time(Timesteps)": settling_time,
        "tolerance_used (per step)": tol_vec.detach().numpy(),
        "y_max_ref": y_max_val,
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
