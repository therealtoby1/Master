import numpy as np
import torch
import torch.optim as optim

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
def extended_kalman_filter(system_fn,jacobian_f,jacobian_h,y_measurements, x0, P0, Q, R, dt, **kwargs):


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
            **kwargs         All arguments that also need to be passed to the system_fn (like K_m u_max)

        Output:
            x_est: estimate of the state'''
    
    n = len(y_measurements)
    x_est = np.zeros((n, 2))
    x = x0.copy()
    x_est[0]=x
    P = P0.copy()

    for k in range(n):
        # === Prediction step ===
        x_pred = system_fn(x, dt,**kwargs)
        F = jacobian_f(x, dt,**kwargs)
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