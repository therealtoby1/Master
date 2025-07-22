import numpy as np

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



################Filter#################################
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

    return x_est

