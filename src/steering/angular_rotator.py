import torch

def adaptive_angular_steering(h, b1, b2, theta):
    """
    Implements Adaptive Angular Steering logic to strictly preserve the L2 activation norm.
    """
    # Compute conditional mask based on the sign of the projection onto d_feat (b1)
    projection_b1 = torch.matmul(h, b1)
    mask = torch.relu(torch.sign(projection_b1)).unsqueeze(-1) 

    # Project the input h onto the 2D plane
    proj_p_h = projection_b1.unsqueeze(-1) * b1 + torch.matmul(h, b2).unsqueeze(-1) * b2
    r = torch.norm(proj_p_h, dim=-1, keepdim=True)

    # Precompute 2D rotation components
    cos_theta = torch.cos(theta).unsqueeze(-1)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    v_theta = cos_theta * b1 + sin_theta * b2

    # Apply the adaptive steering transformation
    h_steered = h + mask * (r * v_theta - proj_p_h) 

    return h_steered
