from filterpy.kalman import KalmanFilter # based on the book Kalman_and_Bayesian_Filters_in_Python.pdf
import numpy as np


class InputParameters:
    """
    Description
    ----------
    An object of Input Parameters for a Kalman Filter.

    Parameters
    ----------
    dim_x : int
      State vector dimension. The default state vector
      dimension is 2, i.e. x_1 = position and x_2 = velocity.
    dim_z : int
      Meassurement vector dimension. The default meassurement
      vector dimension is 1, i.e. z = position.
    x_init : ndarray(dim_x, 1)
      Initial state
    dt : float
      Step size (Delta t)
    F : ndarray(dim_x, dim_x)
      State transition matrix. By default, it assumes constant velocity and that
      the next position is the current position plus current velocity times dt
    H : ndarray(dim_z, dim_x)
      Meassurement function. By default, it assumes that only the position is
      meassured.
    P : ndarray(dim_x, dim_x)
      Covariance matrix describing the uncertainty in the state vector due to
      disturbances that excite the system. By default it is assumed that position
      cannot change suddenly but velocity can change during a time step with a
      variance of 100 (std = 10).
    R : ndarray(dim_z, dim_z)
      Covariance matrix describing the measurement noise. If only position is
      meassured, it can be set as a scalar. By default, it assumes a variance of \
      1 for the noise in the position meassurement (std = 1).
    Q : ndarray(dim_x, dim_x)
      Variance matrix describing the process noise.

    Returns
    ----------
    object
      The Input Parameters for a Kalman Filter.
    """

    def __init__(self, dim_x=2, dim_z=1, x_init=np.array([[0.], [0.]]), dt=.1, F=None, H=np.array([[1., 0.]]),
                 P=np.array([[0., 0.], [0., 0.]]), R=np.array([[1]]),
                 Q=None, Q_process_noise_var=None):
        self.dim_x = dim_x
        self.dim_z = dim_z
        if x_init.shape == (dim_x, 1):
            self.x_init = x_init
        else:
            raise Exception("Initial state must have the same dimension as the state")
        self.dt = dt
        if F is None:
            if dt is not None:
                self.F = np.array([[1., self.dt],
                                   [0., 1.]])
            else:
                raise Exception("If State transition matrix F is not defined, step size"
                                "dt must be defined.")
        else:
            self.F = F
        self.H = H
        self.P = P
        self.R = R
        if Q is None:
            if Q_process_noise_var is None:
                raise Exception("If process noise covariance matrix is not defined, process noise variance "
                                "Q_process_noise_var must be defined.")
            from filterpy.common import Q_discrete_white_noise
            self.Q = Q_discrete_white_noise(dim=dim_x, dt=dt, var=Q_process_noise_var)
        else:
            self.Q = Q


def kf_constvel_smoother(z_vector, dt, measurement_error_std, velocity_std):
    """
    Smooths positions z_vector using a Kalman Filter with a constant velocity model.

    :param z_vector: 1D ndarray containing the signal to smooth
    :param dt: sampling time
    :param measurement_error_std: standard devaiation of measurement error
    :param velocity_std: standard devaiation of velocity
    :return: smoothed version of z_vector
    """
    # Input parameters
    ip = InputParameters(dt=dt, R=measurement_error_std,
                         Q=np.array([[0, 0],
                                     [0, velocity_std**2]]),
                         x_init=np.array([[z_vector[0]], [(z_vector[1]-z_vector[0])/dt]]))

    # Filter definition
    # First, construct the object with the required dimensionality.
    f = KalmanFilter(dim_x=ip.dim_x, dim_z=ip.dim_z)
    # Set initial state
    f.x = ip.x_init
    # Define the state transition matrix
    f.F = ip.F
    # Define the measurement function
    f.H = ip.H
    # Define the covariance matrix that describes the disturbance in the state
    f.P = ip.P
    # Define the covariance matrix that describes the noise in meassurement
    f.R = ip.R
    # Assign the process noise.
    f.Q = ip.Q
    # print(f)

    f.x = ip.x_init
    # Run filter
    # Classical Rauch, Tung, and Striebel
    mu, cov, _, _ = f.batch_filter(z_vector)
    M, P, C, _ = f.rts_smoother(mu, cov)
    x_vector = mu.T.reshape((2, -1))
    x_vector_rts = M.T.reshape((2, -1))

    return x_vector_rts[0, :].reshape((-1,))
