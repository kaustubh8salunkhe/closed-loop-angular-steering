import torch

class LayerPIDController:
    """
    Tracks and updates closed-loop control dynamics across transformer layers.
    """
    def __init__(self, kp: float, ki: float, kd: float, theta_max_deg: float, alpha: float = 1.0, tau: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.theta_max = torch.tensor(theta_max_deg * (torch.pi / 180.0))
        self.alpha = alpha
        self.tau = tau
        
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.layer_history = []

    def reset_state(self):
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.layer_history.clear()

    def compute_steering_angle(self, activation: torch.Tensor, nuisance_vec: torch.Tensor) -> torch.Tensor:
        """
        Computes adaptive rotation angle theta based on instantaneous PID error.
        activation: [B, S, D]
        nuisance_vec: [D]
        """
        # Calculate instantaneous projection error (dot product alignment)
        error = torch.matmul(activation, nuisance_vec)
        mean_error = error.mean()

        # Update PID terms
        p_term = self.kp * mean_error
        self.integral_error += mean_error
        i_term = self.ki * self.integral_error
        d_term = self.kd * (mean_error - self.prev_error)
        self.prev_error = mean_error

        # Control signal u
        u = p_term + i_term + d_term
        
        # Adaptive Sigmoid Gating
        gate = torch.sigmoid(self.alpha * (u - self.tau))
        theta = self.theta_max * gate
        
        return theta
