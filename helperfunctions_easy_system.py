import numpy as np
import torch
import torch.optim as optim
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.means import MultitaskMean, ConstantMean
from gpytorch.kernels import MultitaskKernel, RBFKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.integrate import solve_ivp

def compute_DMD(X,Xprime):
    """
    Computes the Dynamic Mode Decomposition (DMD) of the given data matrices X and Xprime.
    Parameters:
    X (np.array): The data matrix at time t.
    Xprime (np.array): The data matrix at time t+1.
    Returns :
    A (np.array): The DMD matrix that approximates the dynamics of the system so that Xprime = A @ X.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S_inv= np.diag(1/S)
    A = Xprime @ Vt.T @ S_inv @ U.T
    return A, U, Vt, S

def stack_snapshots(X, cols):
    """Stacks snapshots of the state matrix X into a Hankel matrix with a fixed number of columns."""
    n, m = X.shape
    max_delays = m - cols + 1

    if max_delays <= 0:
        raise ValueError(f"Not enough data (m = {m}) to build {cols} columns. "
                         f"Maximum allowed columns is {m - 1}.")

    X_hankel_rows = []

    for i in range(max_delays):
        X_hankel_rows.append(X[:, i : i + cols])

    return np.vstack(X_hankel_rows)

def unstack_snapshots(x_stacked, n):
    """
    Reshapes a np.array into an array with consecutive time
    if x_stacked=[x1,x2]
                 [x2,x3]
    then the output should be [x1,x2,x3], meaning for a dimension 
    of x_stacked (rows,cols) there are cols + (rows-1)*(cols-1) unique timesteps
    
    
    Parameters:
    x_stacked (np.ndarray):  vector of shape (...,cols)for hankel matrix
    n (int): Original system state dimension
    cols (int): Number of time steps (columns) used to form the x_stacked vector
    
    Returns:
    x_matrix (np.ndarray): Reshaped matrix of shape (n, cols) that the "original" states were in
    """
    rows, cols = x_stacked.shape
    Nstacks = rows// n  # number of stacked delay rows

    if rows % n != 0:
        raise ValueError("Number of rows must be divisible by state dimension n.")

    # Start with first block
    x_unstacked = x_stacked[0:n, :].copy()

    # Append the last column from each subsequent stack row
    for i in range(1, Nstacks):
        new_col = x_stacked[i * n:(i + 1) * n, -1][:, np.newaxis]  # shape (n, 1)
        x_unstacked = np.hstack((x_unstacked, new_col))

    return x_unstacked
        

   


def compute_hankel_DMD(X, Xp, cols):
    """
    Constructs a Hankel DMD using a fixed number of columns (snapshots).

    Parameters:
    X (np.array): State matrix at time t, shape (n, m)
    Xp (np.array): State matrix at time t+1, shape (n, m)
    cols : Desired number of column vectors  (time steps used for training)

    returns: 
    see compute_DMD
    """
    stack_snapshots_X = stack_snapshots(X, cols)
    stack_snapshots_Xp = stack_snapshots(Xp, cols)

    return compute_DMD(stack_snapshots_X, stack_snapshots_Xp)



################Filter and observer#################################
def extended_kalman_bucy_filter_fully_continuous(
    system_fn, jacobian_f, jacobian_h, y_measurements, x0, P0, Q, R, dt, D_value=None, **kwargs
):
    """
        Parameters:
            system_fn:      Model of the system that takes current state input x and returns dx/dt
            jacobian_f:     Jacobian of the continuous-time system (f) w.r.t. state
            jacobian_h:     Jacobian of the output function h(x)
            y_measurements: (Noisy) measurements at discrete times (treated as piecewise constant)
            x0:             Initial state estimate
            P0:             Initial error covariance
            Q:              Covariance of process noise
            R:              Covariance of measurement noise
            dt:             Integration step (sampling interval)
            D_value:        Optional sequence of inputs, same length as y_measurements
            **kwargs:       Extra args for system_fn, jacobians

        Returns:
            x_est: (n+1, state_dim) array of state estimates, including initial x0
    """
    n = len(y_measurements)
    state_dim = x0.shape[0]
    x_est = np.zeros((n+1, state_dim))
    x = x0.copy()
    P = P0.copy()

    x_est[0] = x0

    for k in range(n):
        if D_value is not None:
            u_k = D_value[k]
        else:
            u_k = 0

        y_k = y_measurements[k]

        # Joint ODE for state and P (flattened)
        def ekf_ode(t, z):
            # z = [x, P_flat]
            x_ = z[:state_dim]
            P_flat = z[state_dim:]
            P_mat = P_flat.reshape(state_dim, state_dim)

            F = jacobian_f(x_, t, D_value=u_k, **kwargs)
            H = jacobian_h(x_)
            # Correction (Kalman gain)
            K = P_mat @ H.T @ np.linalg.inv(R)
            # Measurement prediction and innovation
            h_x = H @ x_  
            dxdt = system_fn(x_, t, D_value=u_k, **kwargs) + K @ (y_k - h_x)
            dPdt = F @ P_mat + P_mat @ F.T + Q - P_mat @ H.T @ np.linalg.inv(R) @ H @ P_mat
            return np.concatenate([dxdt, dPdt.flatten()])

        # Integrate from t=0 to t=dt using solve_ivp (RK45)
        z0 = np.concatenate([x, P.flatten()])
        sol = solve_ivp(
            ekf_ode, [0, dt], z0, method='RK45', t_eval=[dt]
        )
        z_end = sol.y[:, -1]
        x = z_end[:state_dim]
        P = z_end[state_dim:].reshape(state_dim, state_dim)

        x_est[k+1] = x

    return x_est


def extended_kalman_filter_continous(system_fn, jacobian_f,jacobian_h,y_measurements,x0,P0,Q,R,dt,D_value=0,**kwargs):
    """
        Parameters:
            system_fn:      Model of the system that takes current state input x_k and yields x_(k+1)
            jacobian_f:     Jacobian of the Model (in discrete time!!!)--> F=I+J*dt where J is the Jacobian of the conti-time system
            jacobian_h:     Jacobian of the ouptut function
            y_measurements: (Noisy) Measurements of the output at discrete times t
            x0:              Estimate of the original state
            P0:              Estimate of initial error Covariance
            Q:               Covariance of Process Noise
            R:               Covariance of Measurement-Noise
            dt:              Time between measurement samples
            D_value         Different from UKF here i expect the user (ideally) to pass the whole series of inputs at once.
            **kwargs         All arguments that also need to be passed to the system_fn (like K_m u_max)


        Output:
            x_est: estimate of the state"""
    n = len(y_measurements)
    x_est = np.zeros((n, 2))
    x = x0.copy()
    P = P0.copy()
    state_dim=x0.shape[0]
    for k in range(n):
        if D_value is not None:
            u_k = D_value[k]  #  take the input at step k
        else: u_k = 0
        # === Prediction step ===
        x_pred = system_fn(x, dt, D_value=u_k, **kwargs)
        x_pred = np.asarray(x_pred).flatten()   #  now always (2,). The NN returned a (1,2) with 1 being the nr of batches, which then caused an error in matmul
    
        def P_dot_flat(t, P_flat):
            P_mat = P_flat.reshape(state_dim, state_dim)
            F = jacobian_f(x, dt, D_value=u_k, **kwargs)
            print(F)
            return (F @ P_mat + P_mat @ F.T + Q).flatten()
        
        sol = solve_ivp(P_dot_flat, [0, dt], P.flatten(), method='RK45')
        P_pred = sol.y[:, -1].reshape(state_dim, state_dim)

        # === Update step ===
        H = jacobian_h(x_pred)
        dy = y_measurements[k] - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ dy
        P = (np.eye(2) - K @ H) @ P_pred

        x_est[k] = x

    return np.vstack([x0, x_est])



def extended_kalman_filter_discrete(system_fn, jacobian_f, jacobian_h,
                           y_measurements, x0, P0, Q, R, dt,
                           D_value=0, **kwargs):
    '''
        Parameters:
            system_fn:      Model of the system that takes current state input x_k and yields x_(k+1)
            jacobian_f:     Jacobian of the Model (in discrete time!!!)--> F=I+J*dt where J is the Jacobian of the conti-time system
            jacobian_h:     Jacobian of the ouptut function
            y_measurements: (Noisy) Measurements of the output at discrete times t
            x0:              Estimate of the original state
            P0:              Estimate of initial error Covariance
            Q:               Covariance of Process Noise
            R:               Covariance of Measurement-Noise
            dt:              Time between measurement samples
            D_value         Different from UKF here i expect the user (ideally) to pass the whole series of inputs at once.
            **kwargs         All arguments that also need to be passed to the system_fn (like K_m u_max)


        Output:
            x_est: estimate of the state
        '''
    n = len(y_measurements)
    x_est = np.zeros((n, 2))
    x = x0.copy()
    P = P0.copy()

    for k in range(n):
        if D_value is not None:
            u_k = D_value[k]  #  take the input at step k
        else: u_k = 0
        # === Prediction step ===
        x_pred = system_fn(x, dt, D_value=u_k, **kwargs)
        x_pred = np.asarray(x_pred).flatten()   #  now always (2,). The NN returned a (1,2) with 1 being the nr of batches, which then caused an error in matmul
    
        F = jacobian_f(x, dt, D_value=u_k, **kwargs)
        P_pred = F @ P @ F.T + Q

        # === Update step ===
        H = jacobian_h(x_pred)
        dy = y_measurements[k] - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ dy
        P = (np.eye(2) - K @ H) @ P_pred

        x_est[k] = x

    return np.vstack([x0, x_est])



#==============continous time jacobians for kalman bucy (extended)

def jacobian_biosystem_continous(x, dt, D_value, K_m, s_e, mu_max,**kwargs):
    b, s = x
    rho = mu_max * s / (K_m + s)
    drho_ds = mu_max * K_m / (K_m + s)**2

    df1_db = -D_value + rho
    df1_ds = b * drho_ds

    df2_db = -rho
    df2_ds = -D_value - b * drho_ds

    F = np.array([[df1_db , df1_ds ],
                  [df2_db , df2_ds ]])
    return F

def compute_jacobian_NODE_continuous(x_input, dt=None, D_value=0,**kwargs):
    """Compute discrete-time Jacobian for NODE (F = I + J*dt). returs as a np.ndarray"""
    model = kwargs['model']  # ODEFunc instance
    model.u_seq=D_value
    x_input = torch.tensor(x_input, dtype=torch.float32)
    x = x_input.clone().detach().requires_grad_(True)

    if x.dim() == 1:
        x = x.unsqueeze(0)    # [1, dim_x] this line is not needed anymore (now already handled in the ODEfunc itself!)

    dxdt = model(0, x)        # returns f(x), shape [1, dim_x]
    dxdt = dxdt.view(-1)      # flatten to (dim_x,)

    jacobian = []
    for i in range(dxdt.shape[0]):
        grad = torch.autograd.grad(dxdt[i], x, retain_graph=True)[0]
        grad = grad.squeeze(0)                   #  remove batch dim → (dim_x,) this line is not needed anymore, but doenst affect the code negatively either
        jacobian.append(grad.detach().numpy())

    J = np.stack(jacobian, axis=0)               # shape (dim_x, dim_x)
    return J 
##====Jacobians for the extended Kalman filter

def compute_jacobian_analytical(x, dt, D_value, K_m, s_e, mu_max,**kwargs):
    b, s = x
    rho = mu_max * s / (K_m + s)
    drho_ds = mu_max * K_m / (K_m + s)**2

    df1_db = -D_value + rho
    df1_ds = b * drho_ds

    df2_db = -rho
    df2_ds = -D_value - b * drho_ds

    F = np.array([[1 + df1_db * dt, df1_ds * dt],
                  [df2_db * dt, 1 + df2_ds * dt]])
    return F


def compute_jacobian_autograd(x_input, dt=None, D_value=None, **kwargs):
    model = kwargs['model']   # classicNN_model will be passed here

    # --- ensure state has grad
    x = torch.tensor(x_input, dtype=torch.float32, requires_grad=True)
    if x.dim() == 1:
        x = x.unsqueeze(0)   # -> [1, dim_x]

    # --- ensure D_value is a 2D tensor (no grad)
    u = torch.tensor(D_value, dtype=torch.float32)
    if u.dim() == 0:
        u = u.unsqueeze(0).unsqueeze(1)  # scalar -> [[D]]
    elif u.dim() == 1:
        u = u.unsqueeze(0)               # vector -> [1, dim_u]

    # --- concat along features
    xu = torch.cat([x, u], dim=-1)  # [1, dim_x + dim_u]

    # --- forward pass
    y = model(xu)                   # [1, dim_x]

    # --- flatten output (safe for batch of size 1)
    y = y.view(-1)                  # -> [dim_x]

    # --- compute Jacobian wrt x only
    jacobian = []
    for i in range(y.shape[0]):
        grad_x = torch.autograd.grad(y[i], x, retain_graph=True)[0]  # grad_x is [1, dim_x]
        jacobian.append(grad_x.squeeze(0).detach().numpy())          # squeeze(0) → [dim_x]

    return np.stack(jacobian, axis=0)   # final shape [dim_x, dim_x]

def compute_jacobian_NODE(x_input, dt=None,D_value=0, **kwargs):
    """Compute discrete-time Jacobian for NODE (F = I + J*dt). returs as a np.ndarray"""
    model = kwargs['model']  # ODEFunc instance
    model.u_seq=D_value
    x_input = torch.tensor(x_input, dtype=torch.float32)
    x = x_input.clone().detach().requires_grad_(True)

    if x.dim() == 1:
        x = x.unsqueeze(0)    # [1, dim_x] this line is not needed anymore (now already handled in the ODEfunc itself!)

    dxdt = model(0, x)        # returns f(x), shape [1, dim_x]
    dxdt = dxdt.view(-1)      # flatten to (dim_x,)

    jacobian = []
    for i in range(dxdt.shape[0]):
        grad = torch.autograd.grad(dxdt[i], x, retain_graph=True)[0]
        grad = grad.squeeze(0)                   #  remove batch dim → (dim_x,) this line is not needed anymore, but doenst affect the code negatively either
        jacobian.append(grad.detach().numpy())

    J = np.stack(jacobian, axis=0)               # shape (dim_x, dim_x)
    return np.eye(len(x_input)) + J * dt



def finite_difference_jacobian_gp_bs(x_input,dt=None,D_value=None , eps=1e-2,**kwargs):
    """
    Computes Jacobian of GP output w.r.t. biomass and substrate only.

    Args:
        model: GP model
        likelihood: GP likelihood
        x: [biomass, substrate]
        D: dilution (scalar)
        eps: finite difference step
    Returns:
        jacobian: [n_outputs, 2] numpy array
    """
    model=kwargs['model']
    likelihood=kwargs['likelihood']
    x_in = torch.tensor([[float(x_input[0]), float(x_input[1]), float(D_value)]], dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x_in))
        y0 = pred.mean.squeeze(0)
    n_outputs = y0.shape[0]
    jacobian = np.zeros((n_outputs, 2))

    for i in range(2):  # Only biomass and substrate
        x_plus = x_in.clone()
        x_plus[0, i] += eps
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(x_plus))
            y_plus = pred.mean.squeeze(0)
        jacobian[:, i] = ((y_plus - y0)/eps).numpy()
    return jacobian


def compute_jacobian_h(x):
    # Measurement is only b, so derivative is [1, 0]
    return np.array([[1, 0]])

def mhe_estimate(model, y_meas, u_seq=None, window_size=5,
                 x_init_guess=None, num_iter=10, lr=1e-2,
                 Q_diag=None, R_diag=None, P_diag=None,**kwargs):
    
    """ a function to apply the moving horizon estimation to our measurements. 
    Parameters:
        model:          function that makes x_k-->x_k+1 for the system
        y_meas:         measurements
        u_seq:          input series
        window_size:    nr of measurements used for the moving horizon estimation
        x_init_guess:   initial guess of states (needs to be same size as window size)
        num_iter:       iterations for optimization in each window step
        Q_diag          Process Noise
        R_diag          Measurement Noise
        P_diag          Error Covariance Matrix

    Returns: Estimate of the states up to the last measurement time -window size
    
    
    """

    H = len(y_meas)
    #shifts= how often the sliding window has to iterate to match all the measurements
    nshifts = H - window_size + 1

    # Convert diagonal weights to tensors
    Q_inv = torch.tensor(1.0 / np.array(Q_diag), dtype=torch.float32) if Q_diag is not None else 1.0
    R_inv = torch.tensor(1.0 / np.array(R_diag), dtype=torch.float32) if R_diag is not None else 1.0
    P_inv = torch.tensor(1.0 / np.array(P_diag), dtype=torch.float32) if P_diag is not None else 1.0

    x_est = torch.tensor(x_init_guess, dtype=torch.float32, requires_grad=True)
    estimated_states = []
    prev_estimate = x_est.detach().clone()

    for shift in range(nshifts):
        optimizer = optim.Adam([x_est], lr=lr)

        for _ in range(num_iter):
            optimizer.zero_grad()
            loss = 0.0


            # arrival cost (initial value cost)
            if shift > 0:
                arrival_diff = x_est[0] - prev_estimate[1]
                loss += torch.sum(P_inv * arrival_diff.pow(2))
            

            # measurement cost
            for t in range(window_size):
                y_t = y_meas[shift + t]
                y_pred = x_est[t,0]             #for specific functions this would need modification
                diff = y_t - y_pred
                loss += torch.sum(R_inv * diff.pow(2))
            # model cost
            for t in range(window_size - 1):
                x_t = x_est[t]
                x_pred = model(x_t,**kwargs) if u_seq is None else model(x_t, u_seq[shift + t],**kwargs)

                if isinstance(x_pred, np.ndarray):  #if model returns np. it should be turned into a tensor
                    x_pred = torch.tensor(x_pred, dtype=torch.float32)
    

                diff = x_est[t + 1] - x_pred
                loss += torch.sum(Q_inv * diff.pow(2))

            loss.backward()
            optimizer.step()

        estimated_states.append(x_est[0].detach().clone())
        prev_estimate = x_est.detach().clone()

        with torch.no_grad():
            x_est[:-1] = x_est[1:].clone()
            x_pred = model(x_est[-2], **kwargs)
            if isinstance(x_pred, np.ndarray):
                x_pred = torch.tensor(x_pred, dtype=torch.float32)
            x_est[-1] = x_pred

        # print(shift)
    return torch.stack(estimated_states).numpy()