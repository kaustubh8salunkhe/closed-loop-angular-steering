import torch

def rotate_in_plane(h: torch.Tensor, v_r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Applies a strict norm-preserving 2D rotation to h in the plane spanned by v_r and its orthogonal complement.
    h: [B, S, D]
    v_r: [D] (normalized nuisance vector)
    theta: scalar Tensor (angle in radians)
    """
    # 1. Isolate the component of h along the nuisance vector
    h_proj_r = torch.matmul(h, v_r).unsqueeze(-1) * v_r # [B, S, D]
    
    # 2. Extract orthogonal component
    h_perp = h - h_proj_r
    
    # 3. Create orthonormal basis for the 2D rotation plane (v_r, v_perp_norm)
    r_perp_norm = torch.norm(h_perp, dim=-1, keepdim=True)
    
    # Avoid division by zero for vectors perfectly aligned with v_r
    v_perp_norm = torch.where(r_perp_norm > 1e-8, h_perp / r_perp_norm, torch.zeros_like(h_perp))
    
    # Calculate the in-plane magnitude (which must be preserved)
    # The magnitude in the 2D plane is simply the norm of h itself since we decomposed it this way, 
    # but strictly it is sqrt(||h_proj_r||^2 + ||h_perp||^2) = ||h||
    h_norm = torch.norm(h, dim=-1, keepdim=True)
    
    # 4. Determine initial angle phi in the 2D plane
    # h = h_norm * (cos(phi) * v_r + sin(phi) * v_perp_norm)
    h_proj_r_mag = torch.matmul(h, v_r).unsqueeze(-1)
    phi = torch.atan2(r_perp_norm, h_proj_r_mag)
    
    # 5. Apply the rotation: phi_new = phi + theta
    phi_new = phi + theta
    
    # 6. Reconstruct the vector using the preserved norm
    h_rot = h_norm * (torch.cos(phi_new) * v_r + torch.sin(phi_new) * v_perp_norm)
    
    return h_rot
