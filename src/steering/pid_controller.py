import torch

class LayerPIDController:
    """
    Closed-loop PID Activation Steering across layers.
    """
    def __init__(self, kp: float, ki: float, kd: float, theta_max: float, tau: float = 0.0, alpha: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.theta_max = theta_max
        self.tau = tau
        self.alpha = alpha

        self.integral_error = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral_error = 0.0
        self.prev_error = 0.0

    def compute_theta(self, h, d_feat):
        """
        Computes the rotation angle theta based on instantaneous PID error.
        """
        # Calculate instantaneous projection error (dot product alignment)
        error = torch.matmul(h, d_feat)
        mean_error = error.mean()

        # Compute Proportional, Integral, and Derivative terms
        p_term = self.kp * mean_error
        self.integral_error += mean_error
        i_term = self.ki * self.integral_error
        d_term = self.kd * (mean_error - self.prev_error)

        self.prev_error = mean_error

        # Control signal u(k)
        u_k = p_term + i_term + d_term

        # Adaptive Sigmoid Gating to bound rotation angle theta
        gate = torch.sigmoid(self.alpha * (u_k - self.tau))
        theta = self.theta_max * gate

        return theta
